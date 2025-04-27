import torch
import torch.nn as nn
import torch.optim as optim
import deeplearning as dl
import numpy as np
import pandas as pd
from multiprocessing import freeze_support
import os
import torch.nn.functional as F

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SimpleAdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(SimpleAdaIN, self).__init__()
        self.eps = eps
        
    def calc_mean_std(self, feat):
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True) + self.eps
        return mean, std
    
    def forward(self, content, style):
        # 计算内容和风格的均值和标准差
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)
        
        # 归一化内容特征
        normalized = (content - content_mean) / content_std
        
        # 应用风格统计特性（原始 AdaIN）
        adain_output = normalized * style_std + style_mean
        
        return adain_output

class ChannelEqualizerNet(nn.Module):
    def __init__(self, base_filters=48, num_blocks=2):
        super(ChannelEqualizerNet, self).__init__()
        
        # 简化编码器
        self.signal_encoder = nn.Sequential(
            nn.Conv2d(2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            SimpleResidualBlock(base_filters)
        )
        
        # 简化信道特征提取器
        self.channel_encoder = nn.Sequential(
            nn.Conv2d(2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            SimpleResidualBlock(base_filters)
        )
        
        # 简化AdaIN
        self.transform = SimpleAdaIN()
        
        # 简化解码器
        self.decoder = nn.Sequential(
            SimpleResidualBlock(base_filters),
            nn.Conv2d(base_filters, 2, kernel_size=3, padding=1)
        )
        
    def forward(self, received_signal, channel_matrix):
        # 调整输入维度顺序
        received_signal = received_signal.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        channel_matrix = channel_matrix.permute(0, 3, 1, 2)
        
        # 从接收信号提取特征
        signal_features = self.signal_encoder(received_signal)
        
        # 从信道矩阵提取特征
        channel_features = self.channel_encoder(channel_matrix)
        
        # 使用AdaIN变换信号特征
        transformed = self.transform(signal_features, channel_features)
        
        # 解码变换后的特征
        equalized_signal = self.decoder(transformed)
        
        # 转换回原始形状格式: [batch, H, W, features]
        equalized_signal = equalized_signal.permute(0, 2, 3, 1)
        
        return equalized_signal


def train_channel_equalizer(model, train_loader, epochs=50, learning_rate=0.001, save_dir='models', structure_dir='data/train_frame_structure.json'):
    """
    训练信道均衡网络
    
    Args:
        epochs (int): 训练轮数
        learning_rate (float): 学习率
        batch_size (int): 批量大小
        save_path (str): 模型保存路径
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 加载帧结构配置，用于提取数据符号
    structure_config = pd.read_json(structure_dir)
    frame_structure = structure_config.to_dict()
    symbol_mapping = frame_structure['symbol_mapping']
    symbol_mapping_array = [symbol_mapping[i] for i in sorted(symbol_mapping.keys())]
    symbol_mapping_array = np.array(symbol_mapping_array)
    
    # 找出数据符号的位置索引
    indices_H, indices_W = np.where(symbol_mapping_array == 1)
    
    # 记录最佳损失
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0
        
        for batch_idx, (channel_matrix, received_signal, original_signal) in enumerate(train_loader):
            # 将数据移至设备
            original_signal = original_signal.to(device)
            received_signal = received_signal.to(device)
            channel_matrix = channel_matrix.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(received_signal, channel_matrix)
            
            # 提取数据符号并计算损失
            data_symbols = output[:, indices_H, indices_W, :]
            loss = criterion(data_symbols, original_signal)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印批次训练进度（可选）
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 打印训练损失
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}')
        
        # 保存最佳模型
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f'Best model saved to {best_model_path}')
        
    
    print('Training complete!')
    
    # 返回最终训练好的模型
    return model


def predict_channel_equalizer(model,  channel_matrix, received_signal, structure_dir='data/train_frame_structure.json', checkpoint_path=None):
    """
    使用训练好的信道均衡网络进行预测
    
    Args:
        model: 信道均衡网络模型实例
        test_loader: 测试数据加载器
        structure_dir (str): 帧结构配置文件路径
        checkpoint_path (str, optional): 模型检查点路径，如果提供则加载此模型
    
    Returns:
        tuple: 预测结果和真实标签的元组，以及评估指标
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果提供了检查点路径，则加载模型
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}, trained for {checkpoint['epoch']} epochs")
    
    model = model.to(device)
    
    # 加载帧结构配置，用于提取数据符号
    structure_config = pd.read_json(structure_dir)
    frame_structure = structure_config.to_dict()
    symbol_mapping = frame_structure['symbol_mapping']
    symbol_mapping_array = [symbol_mapping[i] for i in sorted(symbol_mapping.keys())]
    symbol_mapping_array = np.array(symbol_mapping_array)
    
    # 找出数据符号的位置索引
    indices_H, indices_W = np.where(symbol_mapping_array == 1)
    
    # 切换到评估模式
    model.eval()
    
    # 检查输入数据的维度，确保有批次维度
    if len(channel_matrix.shape) == 3:  # [H, W, 2]
        channel_matrix = channel_matrix.unsqueeze(0)  # 增加批次维度 [1, H, W, 2]
    if len(received_signal.shape) == 3:  # [H, W, 2]
        received_signal = received_signal.unsqueeze(0)  # 增加批次维度 [1, H, W, 2]
    
    # 检查输入是否为PyTorch张量，如果不是则转换
    if not isinstance(channel_matrix, torch.Tensor):
        channel_matrix = torch.tensor(channel_matrix, dtype=torch.float32)
    if not isinstance(received_signal, torch.Tensor):
        received_signal = torch.tensor(received_signal, dtype=torch.float32)
    
    # 将数据移至设备
    channel_matrix = channel_matrix.to(device)
    received_signal = received_signal.to(device)
    
    with torch.no_grad():  # 不需要计算梯度
        # 前向传播
        output = model(received_signal, channel_matrix)
        
        batch_size = output.shape[0]
        num_data_symbols = len(indices_H)  # 数据符号的总数
        data_feature_dim = output.shape[-1]  # 特征维度（通常是2，表示实部和虚部）
        
        # 初始化结果张量
        data_symbols = torch.zeros((batch_size, num_data_symbols, data_feature_dim), device=device)
        
        # 使用循环提取每个数据符号位置的值
        for i in range(num_data_symbols):
            h_idx = indices_H[i]
            w_idx = indices_W[i]
            data_symbols[:, i, :] = output[:, h_idx, w_idx, :]
    
    # 返回预测结果（转换为NumPy数组）和数据符号位置索引
    return data_symbols.cpu().numpy()
    