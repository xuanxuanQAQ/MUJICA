import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class CommunicationSignalDataset(Dataset):
    """通信信号与背景噪声分离数据集 - CSV版本"""
    def __init__(self, sample_rate=150, segment_length=4.0, 
                 train_label_dir="data/Train_interpolated_ofdm_signal", 
                 train_input_dir="data/Train_ele_received_phase", 
                 val_label_dir="data/Val_interpolated_ofdm_signal", 
                 val_input_dir="data/Val_ele_received_phase", 
                 split='train'):
        """
        参数:
            sample_rate: 采样率
            segment_length: 信号片段长度（秒）
            train_label_dir: 训练集标签目录
            train_input_dir: 训练集输入目录
            val_label_dir: 验证集标签目录
            val_input_dir: 验证集输入目录
            split: 'train'或'valid'，指定使用训练集还是验证集
        """
        self.data_dir = "data"
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.split = split
        
        # 存储路径
        self.train_label_dir = train_label_dir
        self.train_input_dir = train_input_dir
        self.val_label_dir = val_label_dir
        self.val_input_dir = val_input_dir
        
        # 根据split选择相应的目录
        if split == 'train':
            self.label_dir = self.train_label_dir
            self.input_dir = self.train_input_dir
        elif split == 'valid' or split == 'val':
            self.label_dir = self.val_label_dir
            self.input_dir = self.val_input_dir
        else:
            raise ValueError(f"无效的split值: {split}，必须是'train'或'valid'/'val'")
        
        # 验证目录存在性
        if not os.path.exists(self.label_dir):
            raise ValueError(f"标签目录不存在: {self.label_dir}")
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        # 获取文件列表
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.csv')])
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.csv')])
        
        # 验证文件数量匹配
        if len(self.label_files) != len(self.input_files):
            print(f"警告: 标签文件数量({len(self.label_files)})与输入文件数量({len(self.input_files)})不匹配")
        
        # 验证至少有一个文件
        if not self.label_files or not self.input_files:
            print(f"警告: {split}集中没有找到CSV文件")
            print(f"  标签目录: {self.label_dir}, 文件数: {len(self.label_files)}")
            print(f"  输入目录: {self.input_dir}, 文件数: {len(self.input_files)}")
        
        print(f"初始化{split}数据集完成，共有{len(self.label_files)}个样本")

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.label_files) if hasattr(self, 'label_files') else 0
    
    def __getitem__(self, idx):
        """
        从数据集中获取单个样本
        
        参数:
        idx: 样本索引
        
        返回:
        noisy_signal: 包含噪声的信号 [channels, time]
        clean_signal: 干净的信号 [time, channels]
        """
        # 确保索引在范围内
        if idx >= len(self.label_files):
            raise IndexError(f"索引{idx}超出范围，数据集长度为{len(self.label_files)}")
        
        # 获取文件路径
        noisy_path = os.path.join(self.input_dir, self.input_files[idx])
        clean_path = os.path.join(self.label_dir, self.label_files[idx])
        
        # 确保文件名匹配（可选，检查文件名的数字部分是否一致）
        noisy_id = self.input_files[idx].split('.')[0]
        clean_id = self.label_files[idx].split('.')[0]
        
        if noisy_id != clean_id:
            print(f"警告: 文件ID不匹配 - 输入文件ID: {noisy_id}, 标签文件ID: {clean_id}")
        
        # 加载CSV数据
        try:
            noisy_df = pd.read_csv(noisy_path)
            clean_df = pd.read_csv(clean_path)
        except Exception as e:
            raise ValueError(f"读取CSV文件出错: {e}, noisy_path={noisy_path}, clean_path={clean_path}")
        
        # 检查数据帧是否为空
        if noisy_df.empty or clean_df.empty:
            raise ValueError(f"CSV文件为空: noisy_empty={noisy_df.empty}, clean_empty={clean_df.empty}")
        
        first_col_name = noisy_df.columns[0]
        noisy_signal = torch.tensor(noisy_df[first_col_name].values, dtype=torch.float32)
        
        numeric_cols_clean = clean_df.select_dtypes(include=['number']).columns
        
        clean_signals = []
        for col in numeric_cols_clean:
            clean_signals.append(torch.tensor(clean_df[col].values, dtype=torch.float32))
        
        clean_signal = torch.stack(clean_signals, dim=0)  # [channels, time]
        
        
        # 确保长度匹配
        min_length = min(noisy_signal.shape[0], clean_signal.shape[1])
        noisy_signal = noisy_signal[:min_length]
        clean_signal = clean_signal[:, :min_length]
        
        # 检查形状
        if noisy_signal.dim() != 1:
            raise ValueError(f"噪声信号形状错误，应为一维: noisy_shape={noisy_signal.shape}")
        
        if clean_signal.dim() != 2:
            raise ValueError(f"干净信号形状错误，应为二维: clean_shape={clean_signal.shape}")
        
        # # 转置干净信号以符合模型输入格式 [time, channels]
        # lean_signal = clean_signal.permute(1, 0)  # [time, channels]
        
        # 将干净信号扩展维度以符合 [time, num_spks] 格式
        clean_signal = clean_signal.unsqueeze(-1) if clean_signal.dim() == 1 else clean_signal
        clean_signal = clean_signal.permute(1, 0)  # [time, channels]
        return noisy_signal, clean_signal
    
    