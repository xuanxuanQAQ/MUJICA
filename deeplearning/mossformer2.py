'''Library to support Mossformer2

Authors
* Shengkui Zhao 2024
* Jia Qi Yip 2024
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import PyTorchModelHubMixin
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
import numpy as np
import wandb  # 用于实验追踪（可选）

from .utils.loss import SISDRLoss, CombinedLoss
from .utils.one_path_flash_fsmn import Encoder, Decoder, Dual_Path_Model, SBFLASHBlock_DualA
from .SigSepDataset import CommunicationSignalDataset


def getCheckpoints(config_name):
    
    from huggingface_hub import hf_hub_download

    for file in ['encoder','decoder','masknet']:
        if not os.path.exists(f'./model_weights/{config_name}/{file}.ckpt'):
            print(f'downloading {file}.cpkt')
            hf_hub_download(repo_id=f'alibabasglab/{config_name}', filename=f'{file}.ckpt', local_dir=f'./model_weights/{config_name}')
            print(f'{file}.cpkt downloaded')
        else:
            print(f'{file}.cpkt already downloaded')

class Mossformer2Wrapper(nn.Module, PyTorchModelHubMixin):
    """The wrapper for the Mossformer2 model which combines the Encoder, Masknet and the Encoder
    https://arxiv.org/pdf/2312.11825v1.pdf 

    Example
    -----
    >>> model = Mossformer2Wrapper(config)
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        config: dict
    ):

        super(Mossformer2Wrapper, self).__init__()
        
        self.config_name = config["config_name"]
        print(f'{self.config_name} config loaded')

        self.encoder = Encoder(
            kernel_size=config['encoder_kernel_size'],
            out_channels=config['encoder_out_nchannels'],
            in_channels=config['encoder_in_nchannels'],
        )

        intra_model = SBFLASHBlock_DualA(
            num_layers=config['intra_numlayers'],
            d_model=config['encoder_out_nchannels'],
            nhead=config['intra_nhead'],
            d_ffn=config['intra_dffn'],
            dropout=config['intra_dropout'],
            use_positional_encoding=config['intra_use_positional'],
            norm_before=config['intra_norm_before'],
        )

        self.masknet = Dual_Path_Model(
            in_channels=config['encoder_out_nchannels'],
            out_channels=config['encoder_out_nchannels'],
            intra_model=intra_model,
            num_layers=config['masknet_numlayers'],
            norm=config['masknet_norm'],
            K=config['masknet_chunksize'],
            num_spks=config['masknet_numspks'],
            skip_around_intra=config['masknet_extraskipconnection'],
            linear_layer_after_inter_intra=config['masknet_useextralinearlayer'],
        )
        self.decoder = Decoder(
            in_channels=config['encoder_out_nchannels'],
            out_channels=config['encoder_in_nchannels'],
            kernel_size=config['encoder_kernel_size'],
            stride=config['encoder_kernel_size'] // 2,
            bias=False,
        )
        self.num_spks = config['masknet_numspks']
        self.sample_rate = config['sample_rate']

        # Set device to gpu if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        print(f'model initialised on {self.device}')
    
    @property
    def device(self):
        return next(self.parameters()).device

    def loadPretrained(self):
        if not os.path.isdir(f'./model_weights/{self.config_name}'):
            print("no checkpoints have been cached, getting them now...")
            getCheckpoints(self.config_name)

        #load the model checkpoints
        self.encoder.load_state_dict(torch.load(f'model_weights/{self.config_name}/encoder.ckpt', map_location=torch.device(self.device)))
        self.decoder.load_state_dict(torch.load(f'model_weights/{self.config_name}/decoder.ckpt', map_location=torch.device(self.device)))
        self.masknet.load_state_dict(torch.load(f'model_weights/{self.config_name}/masknet.ckpt', map_location=torch.device(self.device)))
    
    def inference(self, mix_file, output_dir):
        '''
        This is a helper function for inference on a single mixture file
        '''
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        test_mix, sample_rate = torchaudio.load(mix_file)
        
        if sample_rate != self.sample_rate:
            raise Exception(f'Sampling rate must be {self.sample_rate}')
        
        with torch.no_grad():
            est_source = self.forward(test_mix.to(self.device))

        #Normalization to prevent clipping during conversion to .wav file        
        est_source_norm = []
        for ns in range(self.num_spks):
            signal = est_source[0, :, ns]
            signal = signal / signal.abs().max()
            est_source_norm.append(signal.unsqueeze(1).unsqueeze(0))
        est_source = torch.cat(est_source_norm, 2)

        for ns in range(self.num_spks):
            torchaudio.save(
                f'{output_dir}/index{ns+1}.wav', est_source[..., ns].detach().cpu(), sample_rate
            )
        return "done"

    def forward(self, mix):
        """ Processes the input tensor x and returns an output tensor."""
        mix_w = self.encoder(mix)
        if self.config_name == "mossformer2-whamr-2spk":
            est_mask = self.masknet(mix_w)
            sep_h = est_mask
        else:
            est_mask = self.masknet(mix_w)
            mix_w = torch.stack([mix_w] * self.num_spks)
            sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source



def train_mossformer2(config, num_epochs=80, save_dir="model/mossformer2"):
    """训练Mossformer2模型用于通信信号增强 - 简化版本"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = Mossformer2Wrapper(config)
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    
    # 学习率调度器 - 当验证损失停止改善时减小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # 损失函数
    criterion = SISDRLoss()
    
    # 数据加载器
    train_dataset = CommunicationSignalDataset(split='train')
    valid_dataset = CommunicationSignalDataset(split='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True 
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    patience = 10  # 早停耐心值
    best_valid_loss = float('inf')
    no_improve_epochs = 0
    training_history = {
        'train_loss': [],
        'valid_loss': [],
        'learning_rates': []
    }
    
    print(f"开始训练: 训练集大小={len(train_dataset)}, 验证集大小={len(valid_dataset)}")
    
    for epoch in range(num_epochs):
        # =================== 训练阶段 ===================
        model.train()
        epoch_train_losses = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (mixture, sources) in enumerate(pbar):
            # 移动数据到设备
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            est_sources = model(mixture)
            # 计算损失
            loss = criterion(est_sources[:,:,0].unsqueeze(-1), sources[:,:,0].unsqueeze(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            # 优化器步进
            optimizer.step()
            
            # 记录损失
            current_loss = loss.item()
            epoch_train_losses.append(current_loss)
            
            # 更新进度条
            pbar.set_postfix({"loss": current_loss})
        
        # 输出每个epoch的平均损失
        avg_train_loss  = sum(epoch_train_losses) / len(epoch_train_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_train_loss :.6f}")
        
        # =================== 验证阶段 ===================
        model.eval()
        epoch_valid_losses = []
        
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for batch_idx, (mixture, sources) in enumerate(pbar):
                # 移动数据到设备
                mixture = mixture.to(device)
                sources = sources.to(device)
                
                # 前向传播
                est_sources = model(mixture)
                
                # 计算损失
                loss = criterion(est_sources[:,:,0].unsqueeze(-1), sources[:,:,0].unsqueeze(-1))
                
                # 记录损失
                current_loss = loss.item()
                epoch_valid_losses.append(current_loss)
                
                # 更新进度条
                pbar.set_postfix({"loss": current_loss})
        
        # 计算平均验证损失
        avg_valid_loss = sum(epoch_valid_losses) / len(epoch_valid_losses)
        
        # 学习率调度器步进 (基于验证损失)
        scheduler.step(avg_valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['valid_loss'].append(avg_valid_loss)
        training_history['learning_rates'].append(current_lr)
        
        # 输出每个epoch的结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Valid Loss: {avg_valid_loss:.4f}")
        print(f"  Learning rate: {current_lr}")
        
        # 检查是否需要保存模型 (基于验证损失)
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            no_improve_epochs = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': avg_valid_loss,
                'train_loss': avg_train_loss,
                'config': config
            }, best_model_path)
            print(f"  发现更好的模型! 已保存至 {best_model_path}")
        else:
            no_improve_epochs += 1
            print(f"  验证损失未改善. {no_improve_epochs}/{patience}")
        
        # 每N个epoch保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': avg_valid_loss,
                'train_loss': avg_train_loss,
                'config': config
            }, checkpoint_path)
            print(f"  保存检查点至 {checkpoint_path}")
        
        # 绘制训练曲线并保存
        # plot_training_history(training_history, save_dir)
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f"早停触发! {patience}个epoch内验证损失未改善.")
            break
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'valid_loss': avg_valid_loss,
        'train_loss': avg_train_loss,
        'config': config
    }, final_model_path)
    print(f"训练完成! 最终模型已保存至 {final_model_path}")
    
    return model, training_history

def predict_mossformer2(model_path, mix_signal, device=None, config=None):
    """
    使用训练好的Mossformer2模型对混合信号进行分离
    
    参数:
        model_path (str): 模型权重文件的路径
        mix_signal (torch.Tensor or numpy.ndarray): 混合信号，形状为 [time] 或 [batch, time] 或 [batch, time, 1]
        device (torch.device, optional): 推理设备，如果为None则自动选择
        config (dict, optional): 模型配置，如果为None则从模型文件中加载
        
    返回:
        torch.Tensor: 分离后的信号，形状为 [batch, time, num_spks]
    """
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 如果没有提供配置，从模型文件中加载
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("未提供模型配置，且模型文件中也不包含配置信息")
    
    # 初始化模型
    model = Mossformer2Wrapper(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 处理输入数据
    if isinstance(mix_signal, np.ndarray):
        mix_signal = torch.from_numpy(mix_signal).float()
    
    # 确保输入形状正确
    if mix_signal.dim() == 1:  # [time]
        mix_signal = mix_signal.unsqueeze(0)  # [batch, time]
    
    # 移动数据到设备
    mix_signal = mix_signal.to(device)
    
    # 推理
    with torch.no_grad():
        est_sources = model(mix_signal)
    
    return est_sources
    
