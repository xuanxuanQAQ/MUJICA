import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import math
import demodulation

class PositionalEncoding(nn.Module):
    """
    论文'Attention Is All You Need'中描述的位置编码。
    参数:
        d_model (int): 模型/嵌入的维度
        max_len (int, optional): 输入序列的最大长度。默认值为5000。

    方法:
        forward(x): 为输入张量添加位置编码。
            参数:
                x (Tensor): 形状为[batch_size, seq_len, d_model]的输入张量
            返回:
                Tensor: 与位置编码相结合的输入，形状为[batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class Transformer(nn.Module):
    """
    实现联合映射的Transformer模型，适用于信号插值、超采样或类似完型填空的任务
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super().__init__()
        # 输入映射
        self.input_mapping = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器-解码器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        
        # 输出映射
        self.output_mapping = nn.Linear(d_model, input_dim)
        
        # 创建输出位置标记
        self.output_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
    def create_mapping_mask(self, src_len, known_positions, target_len):
        """
        创建掩码来表示已知位置和需要预测的位置
        
        参数:
            src_len: 源序列长度
            known_positions: 已知位置的索引列表
            target_len: 目标序列长度
        """
        # 创建源序列掩码
        src_mask = torch.ones(src_len, src_len, dtype=torch.bool)
        
        # 创建源到目标的注意力掩码
        memory_mask = torch.zeros(target_len, src_len, dtype=torch.bool)
        
        # 创建目标序列掩码 (用于自回归生成)
        tgt_mask = torch.triu(torch.ones(target_len, target_len), diagonal=1).bool()
        
        return src_mask, memory_mask, tgt_mask
    
    def expand_known_points(self, src, known_positions, target_len):
        """
        扩展已知点，创建目标序列的初始猜测
        """
        batch_size = src.size(0)
        device = src.device
        
        # 创建空的目标序列
        tgt = torch.zeros(batch_size, target_len, src.size(-1), device=device)
        
        # 填充已知位置
        for i, pos in enumerate(known_positions):
            if pos < target_len:
                tgt[:, pos, :] = src[:, i, :]
                
        return tgt
    
    def forward(self, src, known_positions, target_len):
        """
        前向传播
        
        参数:
            src: 已知点的输入 [batch_size, num_known_points, input_dim]
            known_positions: 已知点在目标序列中的位置索引
            target_len: 目标序列长度
        """
        # 输入映射和位置编码
        src = self.input_mapping(src)
        src = self.pos_encoder(src)
        
        # 创建初始目标序列
        tgt_init = self.expand_known_points(src, known_positions, target_len)
                
        # 添加位置编码和特殊的输出标记
        tgt = self.pos_encoder(tgt_init)
        
        # 创建掩码
        src_mask, memory_mask, tgt_mask = self.create_mapping_mask(
            src.size(1), known_positions, target_len
        )
        
        device = src.device
        tgt_init = tgt_init.to(device)
        src_mask = src_mask.to(device)
        memory_mask = memory_mask.to(device) 
        tgt_mask = tgt_mask.to(device)
        
        # 通过Transformer
        output = self.transformer(
            src.transpose(0, 1),
            tgt.transpose(0, 1),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # 转换回原始形状并映射到输出维度
        output = output.transpose(0, 1)
        return self.output_mapping(output)

def process_frame_structure(frame_structure):
    if frame_structure is None or 'symbol_mapping' not in frame_structure:
        return None
        
    symbol_mapping = frame_structure['symbol_mapping']
    first_key = next(iter(symbol_mapping))
    seq_len = len(symbol_mapping[first_key])
    
    flattened_map = []
    for slot in sorted(symbol_mapping.keys()):
        flattened_map.extend(symbol_mapping[slot])
    
    map_idx = [i for i, x in enumerate(flattened_map) if x == 2]
    
    total_len = len(flattened_map)
        
    return symbol_mapping, map_idx, total_len


def train_model(model, train_loader, num_epochs, learning_rate=0.001):
    """
    训练Transformer模型
    
    参数:
        model: Transformer模型实例
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    frame_structure = None
    if hasattr(train_loader.dataset, 'frame_structure'):
        frame_structure = train_loader.dataset.frame_structure
    symbol_mapping, map_idx, total_len = process_frame_structure(frame_structure)
    
    if len(train_loader) > 0:
        first_batch = next(iter(train_loader))
        batch_size = first_batch[0].size(0)
    else:
        batch_size = 0
        
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (src, target) in enumerate(train_loader):
               
            src, target = src.to(device), target.to(device)
            
            optimizer.zero_grad()
            channel = model(src, map_idx, total_len)
            
            # 需要把target[idx]改成收到的symbol
            for idx in range(batch_size):
                output = demodulation.apply_equalization(target[idx], channel[idx], symbol_mapping)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        
def predict(model, input_data, known_positions, target_len):
    """
    使用训练好的模型进行预测
    
    参数:
        model: 训练好的Transformer模型
        input_data: 输入数据 [batch_size, num_known_points, input_dim]
        known_positions: 已知点的位置索引
        target_len: 目标序列长度
    返回:
        预测的完整序列
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_data = input_data.to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(input_data, known_positions, target_len)
    
    return predictions.cpu()


if __name__ == '__main__':
    # 测试模型
    model = Transformer(
        input_dim=2,  # 复数信号 (实部+虚部)
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    # 准备数据 - 假设我们有一些采样点
    batch_size = 4
    num_known_points = 10
    target_len = 50  # 我们想要重建的完整序列长度
    
    # 已知点的值 (复数信号用2维表示，实部和虚部)
    src = torch.randn(batch_size, num_known_points, 2)
    
    # 已知点在目标序列中的位置
    known_positions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 均匀分布的点
    
    # 执行插值/超采样
    reconstructed_signal = model(src, known_positions, target_len)
    
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {reconstructed_signal.shape}")