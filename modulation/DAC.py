import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def frequency_shift_real(baseband_signal, time_array, shift_freq, sampling_rate, bandwidth=None):
    """
    对实信号进行频率搬移（余弦调制），并可选择性地应用带通滤波
    
    参数:
    baseband_signal: 输入的基带实信号
    time_array: 对应的时间数组
    shift_freq: 频移量（Hz）
    sampling_rate: 采样率（Hz）
    bandwidth: 带通滤波器带宽（Hz），如果为None则不应用滤波
    
    返回:
    shifted_signal: 频移后的实信号
    """
    # 生成余弦载波
    carrier = np.cos(2 * np.pi * shift_freq * time_array)
    
    # 实数相乘实现频率搬移
    shifted_signal = baseband_signal * carrier
    
    # 如果指定了带宽，则应用带通滤波
    if bandwidth is not None:
        # 设计带通滤波器
        nyquist = sampling_rate / 2
        low = (shift_freq - bandwidth/2) / nyquist
        high = (shift_freq + bandwidth/2) / nyquist
        
        # 确保滤波器参数在有效范围内
        low = max(0.001, min(0.999, low))
        high = max(0.001, min(0.999, high))
        
        # 如果低截止频率低于高截止频率且两者都在有效范围内
        if low < high:
            b, a = signal.butter(4, [low, high], btype='band')
            shifted_signal = signal.filtfilt(b, a, shifted_signal)
        else:
            print(f"警告: 无效的滤波器参数 (low={low}, high={high}), 跳过滤波")
    
    return shifted_signal


def sinc_interpolation(x, original_time, interpolation_factor):
    """
    使用sinc函数进行信号插值
    
    参数:
    x: 原始信号样本
    original_time: 原始采样时间点数组
    interpolation_factor: 插值倍数（整数）
    
    返回:
    interpolated_signal: 插值后的信号
    interpolated_time: 插值后的时间点数组
    time_indices: 每个插值时间点对应的原始采样索引（浮点数）
    """
    # 计算采样间隔
    T = original_time[1] - original_time[0]
    
    # 计算插值后的采样间隔
    T_interp = T / interpolation_factor
    
    # 创建插值时间点数组
    start_time = original_time[0]
    end_time = original_time[-1]
    interpolated_time = np.arange(start_time, end_time + T_interp/2, T_interp)
    
    # 创建输出数组
    interpolated_signal = np.zeros(len(interpolated_time), dtype=complex if np.iscomplexobj(x) else float)
    time_indices = np.zeros(len(interpolated_time))
    
    # 对每个插值时间点计算sinc插值
    for i, t in enumerate(interpolated_time):
        # 计算对应的索引（浮点数）
        idx = (t - original_time[0]) / T
        time_indices[i] = idx
        
        # 应用sinc插值公式
        for n in range(len(x)):
            # sinc函数: sin(π*x)/(π*x)
            sinc_val = np.sinc((t - original_time[n]) / T)
            interpolated_signal[i] += x[n] * sinc_val
    
    return interpolated_signal, interpolated_time, time_indices

def generate_time_indices(input_signal, sampling_rate, start_time=0):
    """
    根据采样率和信号长度生成时间索引数组
    
    参数:
    signal_length: 信号样本数量（整数）
    sampling_rate: 采样率，单位Hz（每秒采样数）
    start_time: 起始时间，默认为0
    
    返回:
    time_indices: 对应每个样本的时间点（秒）
    """
    # 计算采样间隔（秒）
    sampling_interval = 1.0 / sampling_rate
    
    # 生成时间索引数组
    signal_length = len(input_signal)
    time_indices = np.arange(signal_length) * sampling_interval + start_time
    
    return time_indices
