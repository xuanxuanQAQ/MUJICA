import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd


class ChannelDataset(Dataset):
    def __init__(self, signal_dir, pilot_dir, structure_dir, pattern="*.csv"):
        """
        初始化通道数据集
        
        参数:
            signal_dir: 信号数据目录路径
            pilot_dir: 导频数据目录路径
            structure_dir: 帧结构配置文件路径
            pattern: 文件匹配模式
        """
        self.signal_files = sorted(glob.glob(os.path.join(signal_dir, pattern)))
        self.pilot_files = sorted(glob.glob(os.path.join(pilot_dir, pattern)))
        
        # 确保文件数量匹配
        assert len(self.signal_files) == len(self.pilot_files), "信号和导频文件数量不匹配"
        
        # 预加载所有数据，保持每个文件的数据独立
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
                
                # 将numpy数组转为tensor，每个文件作为一个样本
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
        
        print(f"加载了 {len(self.signal_files)} 个文件")
        
        # 读取结构配置文件
        try:
            structure_config = pd.read_json(structure_dir)
            self.frame_structure = structure_config.to_dict()
            print(f"成功加载帧结构配置")
        except Exception as e:
            print(f"读取帧结构配置文件失败: {str(e)}")
            self.frame_structure = None

    def __len__(self):
        return len(self.pilots)

    def __getitem__(self, idx):
        # 返回单个文件的数据作为一个样本
        # 格式为 [length, feature]，其中 feature=2 (实部和虚部)
        return self.pilots[idx], self.signals[idx]
    
def load_channel_data(signal_dir='data/Train_mpsk_signal_B00', 
                      pilot_dir='data/Train_pilot_B00',
                      structure_dir='data/train_frame_structure.json',
                      batch_size=32, 
                      shuffle=True, 
                      num_workers=0):
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
    
    dataset = ChannelDataset(signal_dir, pilot_dir, structure_dir, pattern)
   
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 对于GPU训练有帮助
    )
    
    return data_loader