import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd

# 生成器网络 (Transformer-based)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Generator(nn.Module):
    def __init__(self, N_p=32, N_f=32, N_t=32, d_model=256):
        super(Generator, self).__init__()
        
        self.N_f = N_f
        self.N_t = N_t
        
        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection
        self.output_linear = nn.Linear(d_model, 1)
        self.output_reshape = nn.Linear(N_p, N_f * N_t)

    def forward(self, x):
        # x shape: [batch_size, N_p, 1]
        batch_size = x.size(0)
        
        # Embedding
        x = x.transpose(1, 2)  # [batch_size, 1, N_p]
        x = self.input_embedding(x)  # [batch_size, 1, d_model]
        x = x.transpose(0, 1)  # [1, batch_size, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Output processing
        x = x.transpose(0, 1)  # [batch_size, 1, d_model]
        x = self.output_linear(x)  # [batch_size, 1, 1]
        x = x.squeeze(-1)  # [batch_size, 1]
        x = self.output_reshape(x)  # [batch_size, N_f * N_t]
        x = x.unsqueeze(-1)  # [batch_size, N_f * N_t, 1]
        
        return x
    
# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_gan(generator, discriminator, train_loader, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        for i, (pilot_data, channel_data) in enumerate(train_loader):
            batch_size = pilot_data.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            d_optimizer.zero_grad()
            real_output = discriminator(channel_data)
            d_loss_real = criterion(real_output, real_label)
            
            fake_channel = generator(pilot_data)
            fake_output = discriminator(fake_channel.detach())
            d_loss_fake = criterion(fake_output, fake_label)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_channel)
            g_loss = criterion(fake_output, real_label)
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

def load_channel_data(signal_dir='data/Train_mpsk_signal_B00', 
                      pilot_dir='data/Train_pilot_B00',
                      batch_size=32, 
                      shuffle=True, 
                      num_workers=4):
    """
    加载通道数据
    
    参数:
        signal_dir: 信号数据目录
        pilot_dir: 导频数据目录
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作线程数
    
    返回:
        数据加载器
    """
    # 文件格式检测
    signal_files = glob.glob(os.path.join(signal_dir, "*.*"))
    if not signal_files:
        raise ValueError(f"目录 {signal_dir} 中未找到数据文件")
    
    # 根据实际文件后缀决定匹配模式
    ext = os.path.splitext(signal_files[0])[1]
    pattern = f"*{ext}"
    
    dataset = ChannelDataset(signal_dir, pilot_dir, pattern)
   
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 对于GPU训练有帮助
    )
    
    return data_loader

class ChannelDataset(Dataset):
    def __init__(self, signal_dir, pilot_dir, pattern="*.csv"):
        """
        初始化通道数据集
        
        参数:
            signal_dir: 信号数据目录路径
            pilot_dir: 导频数据目录路径
            pattern: 文件匹配模式
        """
        self.signal_files = sorted(glob.glob(os.path.join(signal_dir, pattern)))
        self.pilot_files = sorted(glob.glob(os.path.join(pilot_dir, pattern)))
        
        # 确保文件数量匹配
        assert len(self.signal_files) == len(self.pilot_files), "信号和导频文件数量不匹配"
        
        # 预加载所有数据
        self.pilots = []
        self.signals = []
        
        for signal_file, pilot_file in zip(self.signal_files, self.pilot_files):
            # 对CSV文件进行处理，读取复数数据
            try:
                # 直接用numpy读取以保持原始行结构
                signal_data = np.loadtxt(signal_file, delimiter=',')
                pilot_data = np.loadtxt(pilot_file, delimiter=',')
                
                # 检查行数是否为偶数(每两行表示一个复数数据)
                if signal_data.shape[0] % 2 != 0 or pilot_data.shape[0] % 2 != 0:
                    print(f"警告: 文件行数不是偶数 - {signal_file}: {signal_data.shape[0]}, {pilot_file}: {pilot_data.shape[0]}")
                
                # 重构复数数据
                # 第一行是实部，第二行是虚部
                signal_complex = signal_data[0] + 1j * signal_data[1]
                pilot_complex = pilot_data[0] + 1j * pilot_data[1]
                
                # 打印复数数组的形状
                print(f"复数数组形状 - 信号: {signal_complex.shape}, 导频: {pilot_complex.shape}")
                
                # 将复数数组转换为双通道实数数组 (实部和虚部分开)
                # shape将从 [cols] 变为 [cols, 2]
                signal_float = np.stack([signal_complex.real, signal_complex.imag], axis=-1).astype(np.float32)
                pilot_float = np.stack([pilot_complex.real, pilot_complex.imag], axis=-1).astype(np.float32)
                
                # 将numpy数组转为tensor
                self.signals.append(torch.FloatTensor(signal_float))
                self.pilots.append(torch.FloatTensor(pilot_float))
                
            except Exception as e:
                print(f"处理文件时出错: {signal_file} 和 {pilot_file}")
                print(f"错误详情: {str(e)}")
                
                # 查看CSV文件内容以进一步诊断
                try:
                    print(f"\n{signal_file} 内容示例:")
                    with open(signal_file, 'r') as f:
                        lines = f.readlines()[:5]  # 读取前5行
                        for i, line in enumerate(lines):
                            print(f"行 {i}: {line.strip()}")
                except:
                    print("无法读取文件内容")
        
        # 合并所有数据
        self.pilot = torch.cat(self.pilots, dim=0)
        self.signal = torch.cat(self.signals, dim=0)
        
        print(f"加载了 {len(self.signal_files)} 个文件，共 {len(self.pilot)} 个样本")

    def __len__(self):
        return len(self.pilot)

    def __getitem__(self, idx):
        return self.pilot[idx], self.signal[idx]

# 使用示例
if __name__ == "__main__":
    # 创建模型
    generator = Generator()
    # discriminator = Discriminator()
    
    train_loader = load_channel_data(
        signal_dir='data/Train_mpsk_signal_B00', 
        pilot_dir='data/Train_pilot_B00',
        batch_size=4
    )
    
    for pilots, signals in train_loader:
        print(f"Batch shapes - Pilots: {pilots.shape}, Signals: {signals.shape}")
        break
    
    # 训练模型
    # train_gan(generator, discriminator, train_loader)