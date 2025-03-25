import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_signals_with_noise(clean_signal, noisy_signals, t, alpha_values, 
                            save_path='alpha_stable_noise.png', show=True):
    """
    绘制原始信号与添加不同alpha值稳定分布噪声后的信号对比图
    
    参数:
    clean_signal: 原始干净信号
    noisy_signals: 添加噪声后的信号列表，每个元素对应一个alpha值
    t: 时间轴数组
    alpha_values: alpha值列表，用于标题显示
    save_path: 保存图像的路径，如果为None则不保存
    show: 是否显示图像，默认为True
    """
    plt.figure(figsize=(15, 10))
    
    for i, (alpha, noisy_signal) in enumerate(zip(alpha_values, noisy_signals)):
        # 绘制信号对比图
        plt.subplot(2, 2, i+1)
        plt.plot(t, clean_signal, 'b-', label='Clean Signal')
        plt.plot(t, noisy_signal, 'r-', alpha=0.7, label='Noisy Signal')
        plt.title(f'Alpha = {alpha}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    if show:
        plt.show(block=False)
    else:
        plt.close()

def plot_noise_distributions(noise_samples, alpha_values,
                            save_path='alpha_stable_distributions.png', show=True):
    """
    绘制不同alpha值稳定分布噪声的概率分布直方图
    
    参数:
    noise_samples: 噪声样本列表，每个元素对应一个alpha值
    alpha_values: alpha值列表，用于标题显示
    save_path: 保存图像的路径，如果为None则不保存
    show: 是否显示图像，默认为True
    """
    plt.figure(figsize=(15, 10))
    
    for i, (alpha, noise) in enumerate(zip(alpha_values, noise_samples)):
        # 绘制分布直方图
        plt.subplot(2, 2, i+1)
        plt.hist(noise, bins=100, density=True, alpha=0.7)
        plt.title(f'Alpha = {alpha} Noise Distribution')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    if show:
        plt.show(block=False)
    else:
        plt.close()
        
def plot_spectrum_and_time_series(w, S, t, eta, U, save_path=None):
    """
    绘制PM频谱和对应的时域波形
    
    参数:
    w: 频率数组 (Hz)
    S: 对应的谱密度数组 (m²/Hz)
    t: 时间数组 (秒)
    eta: 波浪高度数组 (毫米)
    U: 风速 (m/s)，用于标题显示
    save_path: 保存图像的路径，如果为None则不保存
    """
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制频谱
    ax1.plot(w, S, 'b-', linewidth=2)
    ax1.set_xlabel('频率 ω (Hz)', fontsize=14)
    ax1.set_ylabel('谱密度 S(ω) (m²/Hz)', fontsize=14)
    ax1.set_title(f'Pierson-Moskowitz 频谱 (风速 = {U} m/s)', fontsize=16)
    ax1.grid(True)
    
    # 绘制时域波形
    ax2.plot(t, eta, 'r-', linewidth=1.5)
    ax2.set_xlabel('时间 t (秒)', fontsize=14)
    ax2.set_ylabel('波浪高度 η(t) (毫米)', fontsize=14)
    ax2.set_title(f'生成的时域波浪高度 (风速 = {U} m/s)', fontsize=16)
    ax2.grid(True)
    
    # 计算有效波高 (显著波高)
    Hs = 4 * np.sqrt(np.trapz(S, w))
    
    # 添加文本信息
    textstr = f'有效波高 (Hs) = {Hs:.2f} m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图像（如果提供了路径）
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    # 直接显示图像
    plt.show(block=False)
    
    return fig


def plot_microamplitude_wave(t, recieved_signal, U, save_path=None):
    """
    绘制微幅波谱
    
    参数:
    t: 时间数组 (秒)
    eta: 波浪高度数组 (毫米)
    U: 风速 (m/s)，用于标题显示
    save_path: 保存图像的路径，如果为None则不保存
    """

    plt.figure(figsize=(15, 10))

    # 绘制时域波形
    plt.plot(t, recieved_signal, 'r-', linewidth=1.5)
    plt.xlabel('时间 t (秒)', fontsize=14)
    plt.ylabel('波浪高度 η(t) (毫米)', fontsize=14)
    plt.title(f'生成的时域波浪高度 (风速 = {U} m/s)', fontsize=16)
    plt.grid(True)
    
    # 保存图像（如果提供了路径）
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show(block=False)
    