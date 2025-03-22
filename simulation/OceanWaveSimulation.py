import numpy as np
import matplotlib.pyplot as plt

# JONSWAP spectrum function
def JONSWAP(U, X):
    """
    JONSWAP spectrum
    Parameters:
    U: Wind speed (m/s)
    X: Fetch length (m)
    Returns:
    w: Frequency array (Hz)
    S: Spectral density (m^2/Hz)
    """
    w = np.arange(0.01, 1.01, 0.01)  # Frequency range
    S = np.zeros_like(w)
    gamma = 3.3
    g = 9.81  # gravitational acceleration
    
    alpha = 0.076 * (g * X / U**2)**(-0.22)
    omega_p = 22 * (g/U) * (g * X/U**2)**(-0.33)
    
    for i, omega in enumerate(w):
        if omega > omega_p:
            sigma = 0.09
        else:
            sigma = 0.07
            
        K = np.exp(-(omega - omega_p)**2 / (2 * sigma**2 * omega_p**2))
        S[i] = alpha * g**2 / omega**5 * np.exp(-5/4 * (omega_p / omega)**4) * gamma**K
        
    return w, S

# Bretschneider spectrum function
def Bretschneider(U):
    """
    Bretschneider spectrum
    Parameters:
    U: Wind speed (m/s)
    Returns:
    w: Frequency array (Hz)
    S: Spectral density (m^2/Hz)
    """
    w = np.arange(0.0, 1.01, 0.01)  # Frequency range
    S = np.zeros_like(w)
    W_m = 9.81 / U
    H = 0.22 * U**2 / 9.81
    
    for i, omega in enumerate(w):
        if omega == 0:  # Avoid division by zero
            S[i] = 0
        else:
            S[i] = 1.25 / 4 * W_m**4 / omega**5 * H * np.exp(-1.25*(W_m/omega)**4)
            
    return w, S

# Pierson-Moskowitz spectrum function
def PM(U):
    """
    Pierson-Moskowitz spectrum
    Parameters:
    U: Wind speed (m/s)
    Returns:
    w: Frequency array (Hz)
    S: Spectral density (m^2/Hz)
    """
    w = np.arange(0.0, 1.01, 0.01)  # Frequency range
    S = np.zeros_like(w)
    g = 9.81  # gravitational acceleration
    
    for i, omega in enumerate(w):
        if omega == 0:  # Avoid division by zero
            S[i] = 0
        else:
            S[i] = 0.0081 * g**2 / omega**5 * np.exp((-0.74) * (g / (U * omega))**4)
            
    return w, S

def generate_time_series(w, S, duration=20, dt=0.1, random_seed=42):
    """
    从频谱生成时域波浪高度序列
    
    参数:
    w: 频率数组 (Hz)
    S: 频谱密度数组 (m^2/Hz)
    duration: 模拟持续时间 (秒)
    dt: 时间步长 (秒)
    random_seed: 随机数种子，确保结果可重复
    
    返回:
    t: 时间数组 (秒)
    eta: 波浪高度时间序列 (米)
    """
    # 创建时间数组
    t = np.arange(0, duration, dt)
    dw = w[1] - w[0] 

    eta = np.zeros_like(t)
    np.random.seed(random_seed)
    
    # 直接合成 - 对每个频率分量进行叠加
    for i, omega in enumerate(w):
        if S[i] <= 0:
            continue
            
        # 从频谱计算振幅
        a = np.sqrt(2 * S[i] * dw)
        
        # 随机相位 (0到2π)
        phase = np.random.uniform(0, 2*np.pi)
        
        # 将该频率分量添加到总波浪高度
        eta += a * np.cos(2*np.pi * omega * t + phase)
    
    return t, eta


def plot_spectrum_and_time_series(w, S, t, eta, U, save_path=None):
    """
    绘制PM频谱和对应的时域波形
    
    参数:
    w: 频率数组 (Hz)
    S: 对应的谱密度数组 (m²/Hz)
    t: 时间数组 (秒)
    eta: 波浪高度数组 (米)
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
    ax2.set_ylabel('波浪高度 η(t) (米)', fontsize=14)
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
    plt.show()
    
    return fig
