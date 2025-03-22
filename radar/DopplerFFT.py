import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from .RangeFFT import range_fft
from scipy.signal import savgol_filter
from scipy.signal import stft

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


def process_micromotion_phase(data_range_fft):
    """
    处理雷达距离-多普勒图以提取和分析微动分析的相位信息。
    
    参数: 
    data_range_fft : range_fft结果，维度为(num_samples, num_total_chirps)
    
    返回值: 
    processed_phase : 去除异常值后的阈值化微多普勒相位变化
    unwrapped_phase : 原始解缠绕相位时间序列
    smooth_phase :Savitzky-Golay滤波后的平滑相位
    sub_phase : 阈值处理前的微多普勒相位变化
    """
    # 在距离维度上找到能量最大的单元
    max_range_idx = np.argmax(np.sum(np.abs(data_range_fft), axis=1))
    data_range_fft = data_range_fft[max_range_idx, :]

    phase = np.angle(data_range_fft)
    magnitude = np.abs(data_range_fft)

    unwrapped_phase = np.unwrap(phase)

    # 使用Savitzky-Golay滤波器平滑相位
    # 窗口大小和多项式阶数可以根据实际情况调整
    window_size = 51  # 应为奇数
    poly_order = 2
    smooth_phase = savgol_filter(unwrapped_phase, window_size, poly_order)

    # 计算微小相位变化
    sub_phase = unwrapped_phase - smooth_phase

    # 处理异常值
    processed_phase = sub_phase.copy()
    threshold = 0.02  # 根据实际情况调整阈值
    processed_phase[processed_phase > threshold] = 1e-3
    processed_phase[processed_phase < -threshold] = -1e-3

    return processed_phase, unwrapped_phase, smooth_phase, sub_phase


def phase_stft(processed_phase, fs):
    window_size = window_size = min(len(processed_phase) // 2 * 2 - 1, 51)
    noverlap = int(window_size * 0.75)    
    f, t, Zxx = stft(processed_phase, fs=fs, window=signal.windows.kaiser(window_size, 5), 
                        nperseg=window_size, noverlap=noverlap, nfft=len(processed_phase))
    return f, t, Zxx