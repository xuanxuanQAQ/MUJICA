import numpy as np



def simulate_radar_data(num_samples, num_chirps, targets):
    """
    模拟生成毫米波雷达数据
    
    参数:
    num_samples: 每个chirp的采样点数
    num_chirps: chirp的数量
    targets: 目标列表，每个目标是一个元组 (距离，RCS)
    
    返回:
    adc_data: 模拟的ADC数据
    """
    # 创建空的ADC数据数组
    adc_data = np.zeros((num_chirps, num_samples), dtype=complex)
    
    # 雷达参数
    c = 3e8  # 光速 (m/s)
    bandwidth = 4e9  # 4 GHz带宽
    chirp_time = 40e-6  # 40 us chirp时间
    sample_rate = num_samples / chirp_time
    wavelength = c / 77e9  # 77 GHz对应的波长
    
    # 生成时间轴
    t = np.linspace(0, chirp_time, num_samples)
    
    # 为每个目标添加回波
    for target_range, target_rcs in targets:
        # 计算目标的时间延迟
        delay = 2 * target_range / c
        
        # 计算目标的中频信号
        if_freq = 2 * bandwidth * target_range / (c * chirp_time)
        
        # 计算目标回波的相位
        phase = 2 * np.pi * if_freq * t
        
        # 添加目标回波到ADC数据（考虑幅度衰减）
        amplitude = np.sqrt(target_rcs) / (target_range**2)
        for i in range(num_chirps):
            # 添加一些随机相位变化以模拟不同chirp
            random_phase = np.random.uniform(0, 2*np.pi)
            adc_data[i, :] += amplitude * np.exp(1j * (phase + random_phase))
    
    # 添加噪声
    noise_power = 0.01
    noise = np.sqrt(noise_power/2) * (np.random.randn(*adc_data.shape) + 1j * np.random.randn(*adc_data.shape))
    adc_data += noise
    
    return adc_data



# 示例：如何使用上述函数
if __name__ == "__main__":
    # 雷达参数
    num_samples = 512  # 每个chirp的采样点数
    num_chirps = 128   # chirp的数量
    sample_rate = 12.8e6  # 12.8 MHz采样率
    bandwidth = 4e9    # 4 GHz带宽
    center_freq = 77e9  # 77 GHz中心频率
    
    # 设置模拟目标：(距离，雷达散射截面积)
    targets = [
        (5.0, 1.0),   # 5米处的目标
        (10.0, 2.0),  # 10米处的目标
        (15.0, 0.5)   # 15米处的目标
    ]
    
    # 模拟雷达数据
    adc_data = simulate_radar_data(num_samples, num_chirps, targets)
    
    # 执行Range-FFT
    range_profile, range_axis = range_fft(adc_data, num_samples, num_chirps, sample_rate, bandwidth, center_freq)
    
    # 绘制距离谱
    plot_range_profile(range_profile, range_axis)
    
    # 打印检测到的峰值
    avg_profile = np.mean(np.abs(range_profile), axis=0)
    peaks, _ = signal.find_peaks(avg_profile, height=np.max(avg_profile)/10)
    
    print("检测到的目标:")
    for peak in peaks:
        print(f"距离: {range_axis[peak]:.2f} 米, 幅度: {20*np.log10(avg_profile[peak]):.2f} dB")