import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def range_fft(adc_data, num_samples, bandwidth):
    """
    执行毫米波雷达的Range-FFT处理
    
    参数:
    adc_data: 复数形式的ADC原始数据，形状为(num_samples, num_total_chirps)
              其中num_samples是每个chirp的采样点数（通常为256）
              num_total_chirps是所有帧的总chirp数量
    num_samples: 每个chirp的采样点数
    bandwidth: 调频带宽 (Hz)
    
    返回:
    range_fft_result: 距离维度的FFT结果，形状与输入相同
    range_axis: 对应的距离轴 (米)
    """
    # 应用窗函数以降低旁瓣（选择Hanning窗）
    window = np.hanning(adc_data.shape[0])
    # 窗函数应用于第一维（采样点维度）
    windowed_data = adc_data * window[:, np.newaxis]
    
    # 执行Range-FFT (沿采样维度进行FFT)
    range_fft_result = np.fft.fft(windowed_data, n=num_samples, axis=0)
    
    # 计算距离分辨率
    c = 3e8  # 光速 (m/s)
    range_resolution = c / (2 * bandwidth)
    
    # 创建距离轴
    max_range = range_resolution * num_samples
    range_axis = np.linspace(0, max_range, num_samples)
    
    return range_fft_result, range_axis

def plot_range_profile(range_profile, range_axis):
    """
    绘制距离谱
    
    参数:
    range_profile: 距离维度的FFT结果
    range_axis: 对应的距离轴 (米)
    """
    # 计算平均距离谱（跨所有chirps）
    avg_range_profile = np.mean(np.abs(range_profile), axis=0)
    
    # 转换为dB
    range_profile_db = 20 * np.log10(avg_range_profile / np.max(avg_range_profile) + 1e-10)
    
    # 绘制距离谱
    plt.figure(figsize=(10, 6))
    plt.plot(range_axis, range_profile_db)
    plt.grid(True)
    plt.xlabel('距离 (m)')
    plt.ylabel('幅度 (dB)')
    plt.title('毫米波雷达距离谱')
    plt.xlim([0, max(range_axis)])
    plt.ylim([-60, 0])
    plt.show()

