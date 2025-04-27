import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd

class TimeFrequencyFilterNet(nn.Module):
    def __init__(self, hidden_dim=64, n_fft=256, hop_length=64, in_channels=4):
        super(TimeFrequencyFilterNet, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        
        # 时域处理分支 - 修改为支持多通道输入
        self.time_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 频域处理分支 - 对实部和虚部分别处理
        self.freq_encoder_real = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.freq_encoder_imag = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 融合层 - 时域特征映射到频域维度以便融合
        freq_height = n_fft // 2 + 1
        
        # FiLM调制参数生成器 - 从时域特征生成频域特征的调制参数
        self.film_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # 最后一层不需要非线性激活，直接输出调制参数
            nn.AdaptiveAvgPool1d(freq_height) # 自适应池化到频率维度大小
        )
        
        # 分别生成频域实部和虚部的调制系数和偏置
        self.gamma_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=1),
            nn.Tanh() # 使用Tanh可以限制缩放范围，防止数值不稳定
        )
        
        self.beta_generator = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=1)
        )
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 解码器 - 回到频域
        self.decoder_freq = nn.Sequential(
            nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        
        # 时域输出层 - 直接从时域特征生成（输出单通道）
        self.time_out = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, kernel_size=7, padding=3)
        )
        
    def forward(self, x):
        """
        x: [B, C, T] 时域信号，其中C=4为通道数
        """
        batch_size = x.shape[0]
        time_length = x.shape[2]
        
        # 计算STFT后的时间维度
        freq_height = self.n_fft // 2 + 1
        freq_width = (time_length // self.hop_length) + 1
        
        # 时域处理
        time_features = self.time_encoder(x)  # [B, H, T]
        
        # STFT计算 - 需要对每个通道单独计算
        stfts = []
        for i in range(self.in_channels):
            stft_i = torch.stft(
                x[:, i, :], 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                window=torch.hann_window(self.n_fft).to(x.device),
                return_complex=False
            )  # [B, F, T, 2]
            stfts.append(stft_i)
                
        # 堆叠所有通道的STFT结果
        x_stft = torch.stack(stfts, dim=1)  # [B, C, F, T, 2]
        
        # 分离实部和虚部
        x_real = x_stft[..., 0]  # [B, C, F, T]
        x_imag = x_stft[..., 1]  # [B, C, F, T]
        
        # 频域特征提取
        freq_features_real = self.freq_encoder_real(x_real)  # [B, H, F, T]
        freq_features_imag = self.freq_encoder_imag(x_imag)  # [B, H, F, T]
        
        # 组合实部和虚部特征
        freq_features = torch.stack([freq_features_real, freq_features_imag], dim=2)  # [B, H, 2, F, T]
        
        # 生成FiLM调制参数
        film_features = self.film_generator(time_features)  # [B, H, F]
        
        # 生成调制系数gamma和偏置beta
        gammas = self.gamma_generator(film_features)  # [B, H*2, F]
        betas = self.beta_generator(film_features)   # [B, H*2, F]
        
        # 重塑gamma和beta以匹配频域特征维度
        gammas = gammas.view(batch_size, self.hidden_dim, 2, freq_height, 1)  # [B, H, 2, F, 1]
        gammas = gammas.expand(-1, -1, -1, -1, freq_width)                    # [B, H, 2, F, T]
        
        betas = betas.view(batch_size, self.hidden_dim, 2, freq_height, 1)    # [B, H, 2, F, 1]
        betas = betas.expand(-1, -1, -1, -1, freq_width)                      # [B, H, 2, F, T]
        
        # 应用FiLM调制: y = gamma * x + beta
        modulated_features = (1 + gammas) * freq_features + betas  # [B, H, 2, F, T]
        
        # 融合特征处理
        refined_features = self.fusion(modulated_features)  # [B, H, 2, F, T]
        
        # 解码到频域
        output_freq_3d = self.decoder_freq(refined_features)  
        output_freq = output_freq_3d.view(batch_size, 1, 2, freq_height, freq_width)    # [B, 1, 2, F, T]
        
        # 重建为复数形式
        output_freq = output_freq.squeeze(1)  # [B, 2, F, T]
        output_freq = output_freq.permute(0, 2, 3, 1)  # [B, F, T, 2]
        # 将实部和虚部合并为复数张量
        output_freq_complex = torch.complex(
            output_freq[..., 0],  # 实部
            output_freq[..., 1]   # 虚部
        ) 
        # 时域输出
        output_time_direct = self.time_out(time_features)  # [B, 1, T]
        
        padding_needed = (self.hop_length - (output_freq_complex.shape[2] % self.hop_length)) % self.hop_length
        if padding_needed > 0:
            output_freq_complex = F.pad(output_freq_complex, (0, padding_needed))
        else:
            output_freq_complex = output_freq_complex
                
        # 反STFT
        output_time_from_freq = torch.istft(
            output_freq_complex, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            return_complex=False
        )  # [B, T]
        output_time_from_freq = output_time_from_freq.unsqueeze(1)  # [B, 1, T]

        # 正确的维度检查和调整代码
        if output_time_from_freq.shape[2] > output_time_direct.shape[2]:
            output_time_from_freq = output_time_from_freq[:, :, :output_time_direct.shape[2]]
        elif output_time_from_freq.shape[2] < output_time_direct.shape[2]:
            output_time_from_freq = F.pad(output_time_from_freq, (0, output_time_direct.shape[2] - output_time_from_freq.shape[2]))
        
        # 最终输出 - 将直接时域和反变换时域组合（保持单通道输出）
        alpha = 0.7  # 融合参数，可学习
        output = alpha * output_time_direct + (1 - alpha) * output_time_from_freq
        
        return output



class FrequencyAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(FrequencyAttention, self).__init__()
        
        # 注意力网络
        self.attention = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x_stft):
        """
        x_stft: [B, F, T, 2] - 频域复数表示
        """
        # 将输入变换为合适的形状 [B, 2, F, T]
        x_input = x_stft.permute(0, 3, 1, 2)
        
        # 生成注意力图
        attention_map = self.attention(x_input)  # [B, 1, F, T]
        
        # 应用注意力
        x_attended = x_input * attention_map  # [B, 2, F, T]
        
        # 变回原始形状 [B, F, T, 2]
        x_attended = x_attended.permute(0, 2, 3, 1)
        
        return x_attended
    
def train_fusion_denoising(model, train_loader, epochs=50, learning_rate=0.001, save_dir='models'):
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
    
    # 记录最佳损失
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0
        
        for batch_idx, (input, label) in enumerate(train_loader):
            # 将数据移至设备
            input = input.to(device)
            label = label.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(input)
        
            # 提取数据符号并计算损失
            loss = criterion(output, label)
            
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