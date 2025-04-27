import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import radar
import simulation
import demodulation
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from math import ceil
import traceback

class DenoisingExpDataset(Dataset):
    def __init__(self, data_dir, type="BPSK", pattern="*.bin"):
        """
        初始化通道数据集
        
        参数:
            data_dir: 接收信号数据目录
            label_dir: 信号标签数据目录
            mode: 模式（实验-exp 或 仿真-sim）
            pattern: 文件匹配模式

        """
        
        # Parameter setting
        fc = 200  # str2double(fileName(6:8))
        FrameNum = list(range(1, 257))  # 1:256 in MATLAB
        lamda = 3e8 / 77e9
        ChannlNum = 0  # Python is 0-indexed, MATLAB is 1-indexed
        Rb = 100
        fc = 200  # Rb: 码元速率
        modulationIndex = fc / Rb  # Modulate at one bit per two cycles
        
        if type == "BPSK":
            self.recieved_data_files = sorted(glob.glob(os.path.join(data_dir, "BP*SK*" + pattern)))
        elif type == "BFSK":
            self.recieved_data_files = sorted(glob.glob(os.path.join(data_dir, "BF*SK*" + pattern)))
        
        # 预加载所有数据，保持每个文件的数据独立
        self.recieved_datas = []
        self.labels = []
        
        for recieved_data_file in zip(self.recieved_data_files):
            # 对CSV文件进行处理，读取复数数据
            try:
                # 直接用numpy读取以保持原始行结构
                file_path = recieved_data_file[0]
                rawData = radar.read_dca1000(file_path)
    
                params = radar.radar_params_extract(file_path)
                ADCSample, ChirpPeriod, ADCFs, ChirpNum, FramPeriod, FramNum, slope, BandWidth, R_Maximum, R_resulo, V_Maximum, V_resulo = params
                fs = 1e6 / ChirpPeriod

                Len = rawData.shape[1]
                fullChirp = FramPeriod / ChirpPeriod
                
                times, times_compen = radar.create_time_arrays(ChirpPeriod, FrameNum, fullChirp)
                
                frames_dimension = int(round(Len/(ADCSample*ChirpNum)))
                Data_all = np.reshape(rawData, (4, int(ADCSample), int(ChirpNum), frames_dimension), order='F')
                proData = np.reshape(Data_all[:, :, :, np.array(FrameNum)-1], (4, int(ADCSample), -1), order='F')
                
                DataRangeFft, _ = radar.range_fft(proData, int(ADCSample), BandWidth, apply_window=False)
            
                processed_phases = []
                rxDatas = []
                ralocs = []
                for ChannlNum in range(4):    
                    _, maxlocAll = radar.find_max_energy_range_bin(DataRangeFft[ChannlNum, :, :])
                    phase_range = radar.extract_phase_from_max_range_bin(DataRangeFft, maxlocAll, range_search=3, channel_num=ChannlNum, time_increment=1)
                    
                    # Process max power range bin 
                    unwrapped_phase, _ = radar.extract_and_unwrap_phase(phase_range)
                    processed_phase, _, _ = radar.process_micro_phase(unwrapped_phase, times, times_compen, window_size=57, poly_order=3, threshold=0.02)
                    
                    # BPSK demodulation using two methods
                    _, rxData, _, _, raloc = demodulation.bpsk_demodulator_with_symbol_sync(fs, fc, modulationIndex, processed_phase)
                    
                    processed_phases.append(processed_phase)
                    rxDatas.append(rxData)
                    ralocs.append(raloc)
                    
                processed_phases = np.array(processed_phases)    
                
                # print(f'文件: {file_path}')
                
                decoded_signal = utils.majority_vote_decoder(rxDatas)
                # Calculate error rates
                aligned_pattern, error = demodulation.Error110Func(decoded_signal)
                if error > 0.4:
                    print(f"文件 {recieved_data_file} 的误比特率过高: {error}")
                    continue
                
                print(f"File: {recieved_data_file}, Error Rate: {error}")
                
                label, _ = simulation.generate_bpsk_signal(fs, fc, modulationIndex, bit_pattern=aligned_pattern)
                input = utils.trim_input_with_label(processed_phases, label, raloc)
            
                # 将numpy数组转为tensor，每个文件作为一个样本
                self.recieved_datas.append(torch.FloatTensor(input))
                self.labels.append(torch.FloatTensor(label))
                
            except Exception as e:
                print(f"处理文件时出错: {recieved_data_file}")
                print(f"错误详情: {str(e)}")
        
        print(f"成功加载了 {len(self.recieved_datas)} 个文件")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):        
        # 返回单个文件的数据作为一个样本
        return self.recieved_datas[idx], self.labels[idx]
    
    
class ProcessedDenoisingExpDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        """
        初始化数据集
        
        参数:
            input_dir: 输入数据CSV文件所在目录
            label_dir: 标签数据CSV文件所在目录
        """
        
        # 获取所有输入和标签文件
        input_files = sorted(glob.glob(os.path.join(input_dir, '*_input.csv')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*_label.csv')))
        
        # 确保输入和标签文件的数量一致
        assert len(input_files) == len(label_files), "输入文件和标签文件数量不匹配"
        
        # 配对输入和标签文件
        paired_files = []
        for input_file in input_files:
            base_name = os.path.basename(input_file).replace('_input.csv', '')
            label_file = os.path.join(label_dir, f"{base_name}_label.csv")
            if label_file in label_files:
                paired_files.append((input_file, label_file))
        
        # 预加载所有数据，保持每个文件的数据独立
        self.recieved_datas = []
        self.labels = []
        
        for input_file, label_file in paired_files:
            try:
                # 读取输入和标签数据
                input_data = np.loadtxt(input_file, delimiter=',')
                label_data = np.loadtxt(label_file, delimiter=',')
                
                # 确保标签是一维数组
                if label_data.ndim > 1:
                    label_data = label_data.flatten()
                
                # 将numpy数组转为tensor，每个文件作为一个样本
                self.recieved_datas.append(torch.FloatTensor(input_data))
                self.labels.append(torch.FloatTensor(label_data))
                
            except Exception as e:
                print(f"处理文件时出错: {input_file}, {label_file}")
                print(f"错误详情: {str(e)}")
                traceback.print_exc()  # 需要导入traceback模块
        
        print(f"成功加载了 {len(self.recieved_datas)} 个文件")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):        
        # 返回单个文件的数据作为一个样本
        return self.recieved_datas[idx], self.labels[idx]
    
def load_fusion_denoising_exp_data(input_dir='data/exp/bpsk_exp_input',
                      label_dir='data/exp/bpsk_exp_label',
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
    
    dataset = ProcessedDenoisingExpDataset(input_dir=input_dir, label_dir=label_dir)
   
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 对于GPU训练有帮助
    )
    
    return data_loader
