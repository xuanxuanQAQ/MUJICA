import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd


class ChannelEstimationDataset(Dataset):
    def __init__(self, mpsk_label_dir, recieved_signal_dir, structure_dir, pattern="*.csv"):
        """
        初始化通道数据集
        
        参数:
            mpsk_label_dir: MPSK标签路径
            recieved_signal_dir: 接收信号数据目录路径
            structure_dir: 帧结构配置文件路径
            pattern: 文件匹配模式
        """
        self.mpsk_label_files = sorted(glob.glob(os.path.join(mpsk_label_dir, pattern)))
        self.recieved_signal_files = sorted(glob.glob(os.path.join(recieved_signal_dir, pattern)))
        
        # 确保文件数量匹配
        assert len(self.mpsk_label_files) == len(self.recieved_signal_files), "信号和导频文件数量不匹配"
        
        # 预加载所有数据，保持每个文件的数据独立
        self.recieved_signals = []
        self.mpsk_labels = []
        self.pilots = []
        
        # 读取结构配置文件
        try:
            structure_config = pd.read_json(structure_dir)
            self.frame_structure = structure_config.to_dict()
            print(f"成功加载帧结构配置")
        except Exception as e:
            print(f"读取帧结构配置文件失败: {str(e)}")
            self.frame_structure = None
        
        # 获取导频位置的索引
        symbol_mapping = self.frame_structure['symbol_mapping']
        symbol_mapping_array = [symbol_mapping[i] for i in sorted(symbol_mapping.keys())]
        symbol_mapping_array = np.array(symbol_mapping_array)
        pilot_indices = np.where(symbol_mapping_array == 2)
        desired_shape = symbol_mapping_array.shape
        
        for mpsk_label_file, recieved_signal_file in zip(self.mpsk_label_files, self.recieved_signal_files):
            # 对CSV文件进行处理，读取复数数据
            try:
                # 直接用numpy读取以保持原始行结构
                mpsk_label_data = np.loadtxt(mpsk_label_file, delimiter=',')
                recieved_signal_data = np.loadtxt(recieved_signal_file, delimiter=',')
                
                # 检查行数是否为偶数(每两行表示一个复数数据)
                if mpsk_label_data.shape[0] % 2 != 0 or recieved_signal_data.shape[0] % 2 != 0:
                    print(f"警告: 文件行数不是偶数 - {mpsk_label_file}: {mpsk_label_data.shape[0]}, {recieved_signal_file}: {recieved_signal_data.shape[0]}")
                
                # 重构复数数据
                # 第一行是实部，第二行是虚部
                mpsk_label_complex = mpsk_label_data[0] + 1j * mpsk_label_data[1]
                recieved_signal_complex = recieved_signal_data[0] + 1j * recieved_signal_data[1]  
                
                # 将复数数组转换为双通道实数数组 (实部和虚部分开)
                # shape将从 [cols] 变为 [cols, 2]
                mpsk_label_float = np.stack([mpsk_label_complex.real, mpsk_label_complex.imag], axis=-1).astype(np.float32)
                recieved_signal_float = np.stack([recieved_signal_complex.real, recieved_signal_complex.imag], axis=-1).astype(np.float32)
                
                # 将numpy数组转为tensor，每个文件作为一个样本
                self.recieved_signals.append(torch.FloatTensor(recieved_signal_float))
                self.mpsk_labels.append(torch.FloatTensor(mpsk_label_float))
                
                # 获取symbol_mapping_array的大小并重塑recieved_signal_complex
                recieved_signal_complex_shaped = recieved_signal_complex.reshape(desired_shape)
                # 使用导频索引从重塑后的信号中提取导频位置的数据
                pilot_signals = recieved_signal_complex_shaped[pilot_indices]
                # 转换为浮点tensor格式，分离实部和虚部
                pilot_float = np.stack([pilot_signals.real, pilot_signals.imag], axis=-1).astype(np.float32)
                self.pilots.append(torch.FloatTensor(pilot_float))
                
            except Exception as e:
                print(f"处理文件时出错: {mpsk_label_file} 和 {recieved_signal_file}")
                print(f"错误详情: {str(e)}")
                
                # 查看CSV文件内容以进一步诊断
                try:
                    print(f"\n{mpsk_label_file} 内容示例:")
                    with open(mpsk_label_file, 'r') as f:
                        lines = f.readlines()[:5]  # 读取前5行
                        for i, line in enumerate(lines):
                            print(f"行 {i}: {line.strip()}")
                except:
                    print("无法读取文件内容")
        
        print(f"加载了 {len(self.mpsk_label_files)} 个文件")
        

    def __len__(self):
        return len(self.pilots)

    def __getitem__(self, idx):        
        # 返回单个文件的数据作为一个样本
        # 格式为 [length, feature]，其中 feature=2 (实部和虚部)
        return self.pilots[idx], self.recieved_signals[idx], self.mpsk_labels[idx]
    
def load_channel_estimation_data(mpsk_label_dir='data/Train_mpsk_signal_B00', 
                      recieved_signal_dir='data/Train_ofdm_freq_symbols_B00',
                      structure_dir='data/train_frame_structure.json',
                      batch_size=32, 
                      shuffle=True, 
                      num_workers=0):
    """
    加载通道数据
    
    参数:
        mpsk_label_dir: 信号数据目录
        recieved_signal_dir: 接收数据目录
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作线程数
    
    返回:
        数据加载器
    """
    # 文件格式检测
    signal_files = glob.glob(os.path.join(mpsk_label_dir, "*.*"))
    if not signal_files:
        raise ValueError(f"目录 {mpsk_label_dir} 中未找到数据文件")
    
    # 根据实际文件后缀决定匹配模式
    ext = os.path.splitext(signal_files[0])[1]
    pattern = f"*{ext}"
    
    dataset = ChannelEstimationDataset(mpsk_label_dir, recieved_signal_dir, structure_dir, pattern)
   
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 对于GPU训练有帮助
    )
    
    return data_loader