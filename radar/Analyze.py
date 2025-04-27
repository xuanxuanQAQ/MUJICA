import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import stft

def range_fft(adc_data, num_samples, bandwidth, apply_window=True, window_type='hanning'):
    """
    执行毫米波雷达的Range-FFT处理
    
    参数:
    adc_data: 复数形式的ADC原始数据
              可以是(num_samples, num_total_chirps)或
              (channel, num_samples, num_total_chirps)形状
              其中num_samples是每个chirp的采样点数
              num_total_chirps是所有帧的总chirp数量
    num_samples: 每个chirp的采样点数
    bandwidth: 调频带宽 (Hz)
    apply_window: 布尔值，是否应用窗函数，默认为True
    window_type: 字符串，窗函数类型，可选'hanning'、'hamming'、'blackman'等，默认为'hanning'
    
    返回:
    range_fft_result: 距离维度的FFT结果，形状与输入相同
    range_axis: 对应的距离轴 (米)
    """
    # 检查输入维度
    if adc_data.ndim == 2:
        # 如果输入是2维的(num_samples, num_total_chirps)
        if apply_window:
            # 根据指定类型创建窗函数
            if window_type == 'hanning':
                window = np.hanning(adc_data.shape[0])
            elif window_type == 'hamming':
                window = np.hamming(adc_data.shape[0])
            elif window_type == 'blackman':
                window = np.blackman(adc_data.shape[0])
            else:
                # 默认使用Hanning窗
                window = np.hanning(adc_data.shape[0])
            
            # 窗函数应用于第一维（采样点维度）
            windowed_data = adc_data * window[:, np.newaxis]
        else:
            # 不应用窗函数，直接使用原始数据
            windowed_data = adc_data
        
        # 执行Range-FFT (沿采样维度进行FFT)
        range_fft_result = np.fft.fft(windowed_data, n=num_samples, axis=0)
    
    elif adc_data.ndim == 3:
        # 如果输入是3维的(channel, num_samples, num_total_chirps)
        num_channels = adc_data.shape[0]
        
        if apply_window:
            # 根据指定类型创建窗函数
            if window_type == 'hanning':
                window = np.hanning(adc_data.shape[1])
            elif window_type == 'hamming':
                window = np.hamming(adc_data.shape[1])
            elif window_type == 'blackman':
                window = np.blackman(adc_data.shape[1])
            else:
                # 默认使用Hanning窗
                window = np.hanning(adc_data.shape[1])
        
        # 初始化结果数组
        range_fft_result = np.zeros_like(adc_data, dtype=complex)
        
        # 对每个通道分别处理
        for ch in range(num_channels):
            if apply_window:
                # 窗函数应用于第二维（采样点维度）
                windowed_data = adc_data[ch] * window[:, np.newaxis]
            else:
                # 不应用窗函数
                windowed_data = adc_data[ch]
            
            # 执行Range-FFT (沿采样维度进行FFT)
            range_fft_result[ch] = np.fft.fft(windowed_data, n=num_samples, axis=0)
    
    else:
        raise ValueError("输入数据维度必须是2或3")
    
    # 计算距离分辨率
    c = 3e8  # 光速 (m/s)
    range_resolution = c / (2 * bandwidth)
    
    # 创建距离轴
    max_range = range_resolution * num_samples
    range_axis = np.linspace(0, max_range, num_samples)
    
    return range_fft_result, range_axis

def doppler_fft(range_fft_result, num_chirps, chirp_time, fc):
    """
    执行毫米波雷达的Doppler-FFT处理
    
    参数:
    range_fft_result: Range-FFT的结果，形状为(num_samples, num_total_chirps)
                     其中num_samples是距离维的FFT结果数量
                     num_total_chirps是所有帧的总chirp数量
    num_chirps: 每帧的chirp数量（用于多普勒处理）
    chirp_time: 单个chirp的持续时间 (秒)
    fc: 载波频率 (Hz)
    
    返回:
    range_doppler_maps: 三维数组，包含多个二维距离-多普勒图，形状为(num_frames, num_samples, num_chirps)
    doppler_axis: 对应的速度轴 (m/s)
    """
    num_samples = range_fft_result.shape[0]
    num_total_chirps = range_fft_result.shape[1]
    
    # 计算帧数
    num_frames = num_total_chirps // num_chirps
    
    # 创建多普勒窗
    doppler_window = np.hanning(num_chirps)
    
    # 初始化多个距离-多普勒图
    range_doppler_maps = np.zeros((num_frames, num_samples, num_chirps), dtype=complex)
    
    # 对每帧数据分别处理
    for frame_idx in range(num_frames):
        frame_start = frame_idx * num_chirps
        frame_end = (frame_idx + 1) * num_chirps
        
        # 获取当前帧的数据
        frame_data = range_fft_result[:, frame_start:frame_end]  # 形状为(num_samples, num_chirps)
        
        # 对每个距离bin执行Doppler-FFT
        for r in range(num_samples):
            # 提取此距离bin的当前帧所有chirp数据
            range_bin_data = frame_data[r, :]  # 形状为(num_chirps,)
            
            # 应用窗函数
            windowed_data = range_bin_data * doppler_window
            
            # 执行Doppler-FFT
            doppler_fft_result = np.fft.fft(windowed_data)
            
            # 将FFT结果移位，使零频率位于中心
            doppler_fft_result = np.fft.fftshift(doppler_fft_result)
            
            # 存储结果到当前帧的距离-多普勒图
            range_doppler_maps[frame_idx, r, :] = doppler_fft_result
    
    # 计算多普勒分辨率和速度轴
    c = 3e8  # 光速 (m/s)
    lambda_wave = c / fc  # 波长
    prf = 1 / chirp_time  # 脉冲重复频率
    
    # 计算最大可测速度 (由多普勒模糊决定)
    v_max = lambda_wave * prf / 4  # 单向最大速度，除以4是因为双程传播和奈奎斯特采样定理
    
    # 创建速度轴
    doppler_axis = np.linspace(-v_max, v_max, num_chirps)
    
    return range_doppler_maps, doppler_axis

def phase_stft(processed_phase, fs):
    window_size = window_size = min(len(processed_phase) // 2 * 2 - 1, 51)
    noverlap = int(window_size * 0.75)    
    f, t, Zxx = stft(processed_phase, fs=fs, window=signal.windows.kaiser(window_size, 5), 
                        nperseg=window_size, noverlap=noverlap, nfft=len(processed_phase))
    return f, t, Zxx