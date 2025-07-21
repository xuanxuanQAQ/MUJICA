import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from .utils.freq_transform import STDCTLayer
from .utils.flash_fsmn import Encoder, Decoder, SigSepAttnNet, Dual_Flash_FSMN_BLOCK_Wrapper
from .utils.loss import SISDRLoss
from .SigSepDataset import CommunicationSignalDataset

class SignalSeparation(nn.Module):
    
    
    def __init__(self, config):
        super(SignalSeparation, self).__init__()
        
        self.stdct = STDCTLayer(n_fft=config['dct_fft'],
                                hop_length=config['dct_hop_length'],
                                win_length=None,
                                window='hann',
                                center=True,
                                trainable=True)
        self.encoder = Encoder(kernel_size=config['encoder_kernel_size'],
                               out_channels=config['encoder_out_nchannels'],
                               in_channels=config['encoder_in_nchannels'])
        self.decoder = Decoder(kernel_size=config['encoder_kernel_size'],
                               out_channels=config['encoder_in_nchannels'],
                               in_channels=config['encoder_out_nchannels'])

        intra_model = Dual_Flash_FSMN_BLOCK_Wrapper(num_layers=config['intra_numlayers'],
                                                    d_model=config['encoder_out_nchannels'],
                                                    nhead=config['intra_nhead'],
                                                    d_ffn=config['intra_dffn'],
                                                    dropout=config['intra_dropout'],
                                                    use_positional_encoding=config['intra_use_positional'],
                                                    norm_before=config['intra_norm_before'],
                                                )
        
        self.sigattn = SigSepAttnNet(in_channels=config['encoder_out_nchannels'],
                                    out_channels=config['encoder_out_nchannels'],
                                    intra_model=intra_model,
                                    num_layers=config['masknet_numlayers'],
                                    norm=config['masknet_norm'],
                                    skip_around_intra=config['masknet_extraskipconnection'],
                                    linear_layer_after_inter_intra=config['masknet_useextralinearlayer'],
                                    )

    def forward(self, mix):
        
        # [B, T] -> [B, C, F, T]
        freq_mix = self.stdct.forward(mix)
        mix_w, original_shape = self.encoder(freq_mix)
        
        # 噪声分离
        noise_w = self.sigattn(mix_w)
        
        # [B, C, F, T] -> [B, T]
        freq_noise = self.decoder(noise_w, original_shape)
        noise = self.stdct.inverse(freq_noise, mix.shape[-1])
        
        signal = mix - noise
        
        return signal




def train_sigsep(config, num_epochs=80, save_dir="model/mossformer2"):
    """训练Mossformer2模型用于通信信号增强 - 简化版本"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = SignalSeparation(config)
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
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True 
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=8,
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
            signal = model(mixture)
            # 计算损失
            loss = criterion(signal.unsqueeze(-1), sources[:,:,0].unsqueeze(-1))
            
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
                signal = model(mixture)
                # 计算损失
                loss = criterion(signal.unsqueeze(-1), sources[:,:,0].unsqueeze(-1))
                
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
                'train_loss': avg_train_loss
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
                'train_loss': avg_train_loss
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
        'train_loss': avg_train_loss
    }, final_model_path)
    print(f"训练完成! 最终模型已保存至 {final_model_path}")
    
    return model, training_history

        

