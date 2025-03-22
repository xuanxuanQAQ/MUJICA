import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def frequency_demodulate_real(shifted_signal, time_array, shift_freq, sampling_rate, bandwidth=None, lowpass_cutoff=None):
    """
    对经过频率搬移的实信号进行解调，恢复原始基带信号
    
    参数:
    shifted_signal: 频移后的实信号
    time_array: 对应的时间数组
    shift_freq: 原频移量（Hz）
    sampling_rate: 采样率（Hz）
    bandwidth: 可选带通滤波器带宽（Hz），应与调制时相同
    lowpass_cutoff: 低通滤波器截止频率（Hz），用于滤除混频产物，默认为原信号带宽的1.5倍
    
    返回:
    demodulated_signal: 解调后的基带实信号
    """
    # 生成同频同相的余弦载波进行混频
    carrier = np.cos(2 * np.pi * shift_freq * time_array)
    
    # 如果需要，先进行带通滤波以移除可能的干扰
    filtered_signal = shifted_signal
    if bandwidth is not None:
        nyquist = sampling_rate / 2
        low = (shift_freq - bandwidth/2) / nyquist
        high = (shift_freq + bandwidth/2) / nyquist
        
        # 确保滤波器参数在有效范围内
        low = max(0.001, min(0.999, low))
        high = max(0.001, min(0.999, high))
        
        if low < high:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, filtered_signal)
        else:
            print(f"警告: 无效的带通滤波器参数 (low={low}, high={high}), 跳过滤波")
    
    # 与载波相乘进行混频，将信号搬回基带
    # 相乘后会产生基带信号和2倍载波频率的分量
    mixed_signal = filtered_signal * carrier
    
    # 应用低通滤波器去除2倍载波频率分量
    if lowpass_cutoff is None and bandwidth is not None:
        # 如果未指定低通截止频率但有带宽信息，使用带宽的1.5倍作为默认值
        lowpass_cutoff = bandwidth / 2 * 1.5
    elif lowpass_cutoff is None:
        # 如果没有带宽信息，使用采样率的1/4作为默认值
        lowpass_cutoff = sampling_rate / 4
    
    # 设计低通滤波器
    nyquist = sampling_rate / 2
    cutoff = min(0.999, lowpass_cutoff / nyquist)  # 归一化截止频率
    b, a = signal.butter(4, cutoff, btype='low')
    demodulated_signal = signal.filtfilt(b, a, mixed_signal)
    
    # 由于余弦调制会导致信号幅度减半，这里乘以2进行补偿
    # (cos(2πf₁t)·cos(2πf₂t) = 0.5·cos(2π(f₁-f₂)t) + 0.5·cos(2π(f₁+f₂)t))
    demodulated_signal = 2 * demodulated_signal
    
    return demodulated_signal