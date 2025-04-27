import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class TimeFrequencyTransformerNet(nn.Module):
    def __init__(self, hidden_dim=32, n_fft=512, hop_length=128, in_channels=4,      # n_fft应由最高频决定，后续再分析
                 num_encoder_layers=1, num_decoder_layers=1, nhead=8, dropout=0.1,
                 window_size=512, window_stride=256):
        super(TimeFrequencyTransformerNet, self).__init__()
        
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
            nn.ReLU(inplace=True)
        )
        
        # 位置编码 - 为时域序列添加位置信息
        self.time_pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 频域嵌入层 - 对实部和虚部进行嵌入
        self.freq_embedding_real = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.freq_embedding_imag = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 频域位置编码
        self.freq_pos_encoder = Freq2DPositionalEncoding(hidden_dim, dropout)
        
        # FiLM调制参数生成器
        self.film_attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.film_norm = nn.LayerNorm(hidden_dim)
        
        # 生成gamma和beta的变换层
        self.gamma_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.beta_generator = nn.Linear(hidden_dim, hidden_dim)
        
        # Transformer编码器 - 处理调制后的频域特征
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Transformer解码器 - 将频域特征转换回时域
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
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
        
        # 转置为Transformer输入形式 [B, T, H]
        time_features = time_features.transpose(1, 2)
        
        # 添加位置编码
        time_features = self.time_pos_encoder(time_features)
        
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
        
        # 将频域特征重塑为序列形式以便Transformer处理
        freq_features_real = freq_features_real.view(
            batch_size, self.hidden_dim, -1).transpose(1, 2)  # [B, F*T, H]
        
        freq_features_imag = freq_features_imag.view(
            batch_size, self.hidden_dim, -1).transpose(1, 2)  # [B, F*T, H]
        
        # 添加位置编码
        freq_features_real = self.freq_pos_encoder(freq_features_real, 
                                                  self.freq_height, freq_width)
        freq_features_imag = self.freq_pos_encoder(freq_features_imag, 
                                                  self.freq_height, freq_width)
        
        # time_features上采样至freq_features长度
        target_length = freq_features_real.size(1)
        time_features_upsampled = F.interpolate(
            time_features.transpose(1, 2),  # 转为 [B, 64, T]
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2) # 转回 [B, T, 64] 
        
        # 使用时域特征通过注意力机制生成FiLM调制参数
        # 将时域特征作为查询，频域特征作为键和值
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
        
        # 如果序列长度小于窗口大小，直接处理
        if seq_len <= self.window_size:
            encoder_output = self.transformer_encoder(combined_features)
        else:
            # 使用滑动窗口处理长序列
            encoder_output = torch.zeros_like(combined_features)
            counts = torch.zeros(batch_size, seq_len, 1, device=combined_features.device)   
            
            for i in range(0, seq_len - self.window_size + 1, self.window_stride):
                end_idx = min(i + self.window_size, seq_len)
                window_features = combined_features[:, i:end_idx, :]
                window_output = self.transformer_encoder(window_features)
                
                encoder_output[:, i:end_idx, :] += window_output
                counts[:, i:end_idx, :] += 1
                
            encoder_output = encoder_output / counts 
        
        # === 解码器滑动窗口处理 ===
        decoder_seq_len = time_features.size(1)
        counts = []
        
        # 如果序列长度小于窗口大小，直接处理
        if decoder_seq_len <= self.window_size:
            decoder_output = self.transformer_decoder(time_features, encoder_output)
        else:
            # 使用滑动窗口处理长序列
            decoder_output = torch.zeros_like(time_features)
            counts = torch.zeros(batch_size, decoder_seq_len, 1, device=time_features.device)
            
            for i in range(0, decoder_seq_len - self.window_size + 1, self.window_stride):
                end_idx = min(i + self.window_size, decoder_seq_len)
                
                # 从解码器序列中获取窗口
                window_decoder_input = time_features[:, i:end_idx, :]
                
                # 使用完整的编码器输出（或者可以也对编码器输出进行窗口化处理）
                window_decoder_output = self.transformer_decoder(
                    window_decoder_input, encoder_output
                )
                
                # 在线累加窗口输出到最终结果
                decoder_output[:, i:end_idx, :] += window_decoder_output
                counts[:, i:end_idx, :] += 1
                
            # 计算平均值（处理重叠区域）
            decoder_output = decoder_output / counts 
    
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


# 位置编码器 - 用于Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: [B, S, E] Batch, Sequence, Embedding
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


# 二维频域位置编码器 - 为频域特征添加位置信息
class Freq2DPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200000):
        super(Freq2DPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建足够长的位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe_1d = torch.zeros(max_len, d_model)
        pe_1d[:, 0::2] = torch.sin(position * div_term)
        pe_1d[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_1d', pe_1d)
        
    def forward(self, x, freq_dim, time_dim):
        """
        x: [B, F*T, E] - 已经展平的频域特征
        freq_dim: 频率维度大小
        time_dim: 时间维度大小
        """
        batch_size, _, d_model = x.shape
        
        # 为每个频率和时间位置创建唯一的编码
        pe_f = self.pe_1d[:freq_dim]  # 频率位置编码
        pe_t = self.pe_1d[:time_dim]  # 时间位置编码
        
        # 创建2D位置编码
        pe_2d = torch.zeros(freq_dim * time_dim, d_model, device=x.device)
        
        # 填充2D位置编码
        for f in range(freq_dim):
            for t in range(time_dim):
                idx = f * time_dim + t
                # 组合频率和时间位置编码
                pe_2d[idx] = pe_f[f] + pe_t[t]
        
        # 添加位置编码到输入
        x = x + pe_2d.unsqueeze(0)
        return self.dropout(x)
    
    
def train_fusion_denoising_transformer(model, train_loader, epochs=50, learning_rate=0.001, save_dir='models'):
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