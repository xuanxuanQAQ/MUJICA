import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from numpy.linalg import inv





def joint_power_modulation_optimization(channel_gains, total_power, target_ber=1e-3):
    # 可用的调制方案及其频谱效率(bits/symbol)
    modulation_schemes = {'BPSK': 1, 'QPSK': 2, '16QAM': 4, '64QAM': 6}
    
    # 各调制方案达到目标BER所需最小SNR
    required_snr = {'BPSK': 6.8, 'QPSK': 9.8, '16QAM': 16.5, '64QAM': 22.5}  # dB
    
    # 初始化结果
    allocation = []
    
    # 根据信道增益排序子载波
    sorted_indices = np.argsort(-channel_gains)  # 降序
    
    # 贪婪分配算法
    remaining_power = total_power
    for idx in sorted_indices:
        gain = channel_gains[idx]
        
        # 选择可行的最高阶调制方案
        selected_mod = 'BPSK'  # 默认最低阶
        selected_power = 0
        best_efficiency = 0
        
        for mod, bits in modulation_schemes.items():
            # 计算所需功率
            required_power = (10**(required_snr[mod]/10)) / gain
            
            if required_power <= remaining_power:
                efficiency = bits / required_power
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    selected_mod = mod
                    selected_power = required_power
        
        if selected_power > 0:
            remaining_power -= selected_power
            allocation.append((idx, selected_mod, selected_power))
    
    return allocation