import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class TimeFrequencyBiGRUNet(nn.Module):
    def __init__(self, hidden_dim=32, n_fft=256, hop_length=64, in_channels=4,
                 num_gru_layers=1, dropout=0.1, nhead=4,
                 window_size=512, window_stride=256):
        super(TimeFrequencyBiGRUNet, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        
        self.window_size = window_size
        self.window_stride = window_stride
        
        # 频域维度
        self.freq_height = n_fft // 2 + 1
        
        # 时域嵌入层 - 将多通道时域信号转换为特征嵌入
        self.time_embedding = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 频域嵌入层 - 对实部和虚部进行嵌入
        self.freq_embedding_real = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.freq_embedding_imag = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # FiLM调制参数生成器
        self.film_attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.film_norm = nn.LayerNorm(hidden_dim)
        
        self.film_projection = nn.Linear(hidden_dim * 2, hidden_dim)  # 投影回原始维度
        
        # 生成gamma和beta的变换层
        self.gamma_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.beta_generator = nn.Linear(hidden_dim, hidden_dim)
        
        # 替代Transformer编码器的双向GRU
        self.encoder_bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # 编码器输出投影层 - 将BiGRU的输出投影回原始维度
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 替代Transformer解码器的双向GRU
        self.decoder_bigru = nn.GRU(
            input_size=hidden_dim * 2,  # 输入维度包括时域特征和编码器输出的拼接
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # 解码器输出投影层
        self.decoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 频域输出投影
        self.freq_projection = nn.Linear(hidden_dim, 2)  # 实部和虚部
        
        # 时域输出层
        self.time_projection = nn.Linear(hidden_dim, 1)
        
        # 可学习的融合参数
        self.alpha = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, x):
        """
        x: [B, C, T] 时域信号，其中C=4为通道数
        """
        batch_size, _, time_length = x.shape
        
        # 计算STFT后的时间维度
        freq_width = (time_length // self.hop_length) + 1
        
        # 时域特征提取
        time_features = self.time_embedding(x)  # [B, H, T]
        
        # 转置为序列处理形式 [B, T, H]
        time_features = time_features.transpose(1, 2)
        
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
        
        # 频域特征嵌入
        freq_features_real = self.freq_embedding_real(x_real)  # [B, H, F, T]
        freq_features_imag = self.freq_embedding_imag(x_imag)  # [B, H, F, T]
        
        # 将频域特征重塑为序列形式以便GRU处理
        freq_features_real = freq_features_real.view(
            batch_size, self.hidden_dim, -1).transpose(1, 2)  # [B, F*T, H]
        
        freq_features_imag = freq_features_imag.view(
            batch_size, self.hidden_dim, -1).transpose(1, 2)  # [B, F*T, H]
        
        # time_features上采样至freq_features长度
        target_length = freq_features_real.size(1)
        time_features_upsampled = F.interpolate(
            time_features.transpose(1, 2),  # 转为 [B, H, T]
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # 转回 [B, T, H] 
        
        # 生成FiLM调制参数
        attn_output_real, _ = self.film_attention(
            time_features_upsampled, freq_features_real, freq_features_real
        )  # [B, T, H]
        
        attn_output_imag, _ = self.film_attention(
            time_features_upsampled, freq_features_imag, freq_features_imag
        )  # [B, T, H]
        
        # 残差连接和归一化
        film_features_real = self.film_norm(time_features_upsampled + attn_output_real)  # [B, T, H]
        film_features_imag = self.film_norm(time_features_upsampled + attn_output_imag)  # [B, T, H]
        
        
        # 生成gamma和beta
        gamma_real = self.gamma_generator(film_features_real)  # [B, T, H]
        beta_real = self.beta_generator(film_features_real)    # [B, T, H]
        
        gamma_imag = self.gamma_generator(film_features_imag)  # [B, T, H]
        beta_imag = self.beta_generator(film_features_imag)    # [B, T, H]
        
        
        # 应用FiLM调制
        modulated_real = (1 + gamma_real) * freq_features_real + beta_real
        modulated_imag = (1 + gamma_imag) * freq_features_imag + beta_imag
        
        # 合并调制后的实部和虚部特征
        # 使用交替方式组合序列，以保持实部和虚部信息
        seq_len = modulated_real.size(1)
        combined_features = torch.zeros(
            batch_size, seq_len*2, self.hidden_dim, device=x.device
        )
        combined_features[:, 0::2, :] = modulated_real
        combined_features[:, 1::2, :] = modulated_imag
        
        # 处理编码器输入 - 使用滑动窗口处理长序列
        seq_len = combined_features.size(1)
        
        # 滑动窗口处理函数，用于处理长序列
        def process_with_sliding_windows(features, gru_model, projection_layer=None):
            seq_len = features.size(1)
            
            # 如果序列长度小于窗口大小，直接处理
            if seq_len <= self.window_size:
                output, _ = gru_model(features)
                if projection_layer is not None:
                    output = projection_layer(output)
                return output
            
            # 使用滑动窗口处理长序列
            batch_size, _, hidden_dim = features.shape
            output_dim = hidden_dim if projection_layer is None else projection_layer.out_features
            
            output = torch.zeros(batch_size, seq_len, output_dim, device=features.device)
            counts = torch.zeros(batch_size, seq_len, 1, device=features.device)
            
            for i in range(0, seq_len - self.window_size + 1, self.window_stride):
                end_idx = min(i + self.window_size, seq_len)
                window_features = features[:, i:end_idx, :]
                
                window_output, _ = gru_model(window_features)
                
                if projection_layer is not None:
                    window_output = projection_layer(window_output)
                
                output[:, i:end_idx, :] += window_output
                counts[:, i:end_idx, :] += 1
            
            # 处理可能的边界窗口
            if seq_len % self.window_stride != 0:
                i = max(0, seq_len - self.window_size)
                window_features = features[:, i:, :]
                window_output, _ = gru_model(window_features)
                
                if projection_layer is not None:
                    window_output = projection_layer(window_output)
                
                output[:, i:, :] += window_output
                counts[:, i:, :] += 1
            
            # 计算平均值（处理重叠区域）
            output = output / counts
            
            return output
        
        # 使用BiGRU编码器处理特征
        encoder_output = process_with_sliding_windows(
            combined_features, self.encoder_bigru, self.encoder_projection
        )
        
        # 准备解码器输入 - 拼接时域特征和编码器输出
        # 首先调整编码器输出的大小以匹配时域特征
        encoder_output_resized = F.interpolate(
            encoder_output.transpose(1, 2),  # [B, H, T']
            size=time_features.size(1),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [B, T, H]
        
        # 拼接时域特征和编码器输出
        decoder_input = torch.cat([time_features, encoder_output_resized], dim=2)  # [B, T, H*2]
        
        # 使用BiGRU解码器处理
        decoder_output = process_with_sliding_windows(
            decoder_input, self.decoder_bigru, self.decoder_projection
        )
        
        # 生成频域输出 - 用于反STFT
        freq_output = self.freq_projection(encoder_output)  # [B, F*T*2, 2]
        
        # 将频域输出重新组织为STFT格式
        # 分离实部和虚部
        freq_real = freq_output[:, 0::2, 0].view(batch_size, self.freq_height, freq_width)
        freq_imag = freq_output[:, 1::2, 1].view(batch_size, self.freq_height, freq_width)
        
        # 创建复数STFT表示
        output_freq_complex = torch.complex(freq_real, freq_imag)  # [B, F, T]
        
        # 时域直接输出
        output_time_direct = self.time_projection(decoder_output).squeeze(-1)  # [B, T]
        output_time_direct = output_time_direct.unsqueeze(1)  # [B, 1, T]
        
        # 确保频域输出长度合适
        padding_needed = (self.hop_length - (output_freq_complex.shape[-1] % self.hop_length)) % self.hop_length
        if padding_needed > 0:
            output_freq_complex = F.pad(output_freq_complex, (0, padding_needed))
        
        # 反STFT
        output_time_from_freq = torch.istft(
            output_freq_complex, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            return_complex=False
        )  # [B, T]
        output_time_from_freq = output_time_from_freq.unsqueeze(1)  # [B, 1, T]
        
        # 调整输出维度以便融合
        if output_time_from_freq.shape[2] > output_time_direct.shape[2]:
            output_time_from_freq = output_time_from_freq[:, :, :output_time_direct.shape[2]]
        elif output_time_from_freq.shape[2] < output_time_direct.shape[2]:
            output_time_from_freq = F.pad(output_time_from_freq, (0, output_time_direct.shape[2] - output_time_from_freq.shape[2]))
        
        # 最终输出 - 加权融合
        output = self.alpha * output_time_direct + (1 - self.alpha) * output_time_from_freq
        
        return output


# 移除了位置编码相关类，GRU模型不需要这些
    
    
def train_fusion_denoising_bigru(model, train_loader, epochs=50, learning_rate=0.001, save_dir='models'):
    """
    训练BiGRU信道均衡网络
    
    Args:
        model: BiGRU模型
        train_loader: 训练数据加载器
        epochs (int): 训练轮数
        learning_rate (float): 学习率
        save_dir (str): 模型保存目录
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
            
            # 计算损失
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
            best_model_path = os.path.join(save_dir, 'best_bigru_model.pth')
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