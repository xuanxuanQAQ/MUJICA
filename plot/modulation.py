import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_constellation(symbols, title):
    """绘制星座图"""
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(symbols), np.imag(symbols), c='b', marker='.')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(title)
    plt.axis('equal')
    plt.show(block=False)

def plot_ofdm_signal_analysis(ofdm_signal, n_fft, sample_rate, time_title="OFDM信号时域", freq_title="OFDM信号频谱"):
    """
    绘制OFDM信号的时域和频域表示
    
    参数:
    ofdm_signal -- OFDM信号数据
    n_fft -- FFT大小
    sample_rate -- 采样率
    time_title -- 时域图标题
    freq_title -- 频域图标题
    """
    # 创建包含两个子图的图形
    plt.figure(figsize=(12, 10))
    
    # 时域图 (上图)
    plt.subplot(2, 1, 1)
    time = np.arange(len(ofdm_signal)) / sample_rate
    plt.plot(time, np.real(ofdm_signal), 'b-', label='实部')
    plt.plot(time, np.imag(ofdm_signal), 'r-', label='虚部')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.title(time_title)
    plt.legend()
    
    # 频域图 (下图)
    plt.subplot(2, 1, 2)
    # 使用welch方法计算功率谱密度
    f, Pxx = signal.welch(ofdm_signal, sample_rate, nperseg=n_fft, nfft=n_fft*4, 
                        scaling='density', return_onesided=True)
    
    positive_mask = f >= 0
    f = f[positive_mask]
    Pxx = Pxx[positive_mask]
    
    # 转换为dB单位
    Pxx_db = 10 * np.log10(Pxx + 1e-10)  # 添加小值避免log(0)
    
    # 绘制单边频谱
    plt.plot(f, Pxx_db)
    plt.grid(True)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title(freq_title)
    
    # 设置x轴范围 - 只显示从0到奈奎斯特频率
    plt.xlim(0, sample_rate/2)
    
    # 添加OFDM子载波位置标记（如果子载波数量不太多）
    if n_fft <= 64:
        subcarrier_spacing = sample_rate / n_fft
        subcarrier_positions = np.arange(0, sample_rate/2, subcarrier_spacing)
        
        for pos in subcarrier_positions:
            plt.axvline(x=pos, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
def plot_signal_with_time_indices(signal_data, time_indices, title="Signal Analysis", figsize=(12, 8), auto_center=True):
    """
    根据信号时域数据和时间索引绘制时域谱和频域谱，自动检测频谱中心
    
    参数:
    signal_data: 时域信号数据
    time_indices: 对应的时间索引数组
    title: 图表标题
    figsize: 图表尺寸
    auto_center: 是否自动检测并以频谱峰值为中心显示
    
    返回:
    fig: matplotlib图表对象
    """
    # 检查输入数据
    if len(signal_data) != len(time_indices):
        raise ValueError(f"信号数据长度({len(signal_data)})与时间索引长度({len(time_indices)})不匹配")
    
    # 计算采样率
    if len(time_indices) > 1:
        sampling_intervals = np.diff(time_indices)
        avg_interval = np.mean(sampling_intervals)
        fs = 1.0 / avg_interval
    else:
        raise ValueError("至少需要两个时间点来计算采样率")
    
    # 创建图表
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    
    # 绘制时域信号
    axs[0].plot(time_indices, signal_data, 'b-')
    axs[0].set_title('时域信号')
    axs[0].set_xlabel('时间 (秒)')
    axs[0].set_ylabel('幅度')
    axs[0].grid(True)
    
    # 计算频谱
    freq_data = np.fft.fft(signal_data)
    freq_axis = np.fft.fftfreq(len(signal_data), avg_interval)
    
    # 计算功率谱密度
    psd = np.abs(freq_data)**2 / len(signal_data)
    
    # 如果启用自动中心检测
    detected_center = 0
    if auto_center:
        # 找到频谱峰值的位置
        peak_idx = np.argmax(psd)
        detected_center = freq_axis[peak_idx]
        
        # 如果峰值在负频率区域，可能需要考虑共轭对称性
        if detected_center < 0 and np.iscomplexobj(signal_data):
            # 复数信号可能有不对称的频谱
            pass
        elif detected_center < 0:
            # 实数信号的频谱是共轭对称的，可以使用正频率部分
            positive_peak_idx = np.argmax(psd[len(psd)//2:]) + len(psd)//2
            detected_center = freq_axis[positive_peak_idx]
    
    # 绘制频域信号 (以dB为单位)
    psd_db = 10 * np.log10(psd + 1e-10)  # 加一个小数防止log(0)
    
    # 对频率和PSD进行排序以便正确绘图
    sort_idx = np.argsort(freq_axis)
    freq_axis_sorted = freq_axis[sort_idx]
    psd_db_sorted = psd_db[sort_idx]
    
    axs[1].plot(freq_axis_sorted, psd_db_sorted, 'r-')
    
    if auto_center:
        axs[1].set_title(f'频域信号 (功率谱密度) - 检测到的中心频率: {detected_center:.2f} Hz')
        
        # 设置频率轴范围，使检测到的中心频率居中显示
        freq_range = fs / 2  # 奈奎斯特频率
        axs[1].set_xlim([detected_center - freq_range/2, detected_center + freq_range/2])
    else:
        axs[1].set_title('频域信号 (功率谱密度)')
    
    axs[1].set_xlabel('频率 (Hz)')
    axs[1].set_ylabel('功率/频率 (dB/Hz)')
    axs[1].grid(True)
    
    # 设置整体标题
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, detected_center