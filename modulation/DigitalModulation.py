import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def mpsk_modulation(bits, M):
    """
    实现MPSK调制
    
    参数:
    bits -- 二进制比特流(0和1组成的列表或数组)
    M -- 相位数，必须是2的幂(如2, 4, 8, 16等)
    
    返回:
    complex_symbols -- 复数符号点
    """
    if not (M & (M-1) == 0) or M < 2:
        raise ValueError("M必须是2的幂，如2, 4, 8, 16等")
    
    # 每个符号的比特数
    bits_per_symbol = int(np.log2(M))
    
    # 确保比特长度是bits_per_symbol的整数倍
    if len(bits) % bits_per_symbol != 0:
        padding = bits_per_symbol - (len(bits) % bits_per_symbol)
        bits = np.pad(bits, (0, padding), 'constant')
    
    # 将bits分组
    bits_groups = bits.reshape(-1, bits_per_symbol)
    
    # 转换每组比特为十进制
    decimal_values = np.zeros(len(bits_groups))
    for i in range(bits_per_symbol):
        decimal_values += bits_groups[:, i] * (2 ** (bits_per_symbol - 1 - i))
    
    # 计算相位角度，转换为复数符号
    phase_angles = decimal_values * (2 * np.pi / M) 
    complex_symbols = np.exp(1j * phase_angles)
    
    return complex_symbols

def qam_modulation(bits, M):
    """
    实现QAM(Quadrature Amplitude Modulation)调制
    
    参数:
    bits -- 二进制比特流(0和1组成的列表或数组)
    M -- 调制阶数，必须是平方数(如4, 16, 64, 256等)
    
    返回:
    complex_symbols -- 复数符号点
    """
    # 检查M是否为平方数
    K = int(np.sqrt(M))
    if K * K != M:
        raise ValueError("M必须是平方数，如4, 16, 64, 256等")
    
    # 每个符号的比特数
    bits_per_symbol = int(np.log2(M))
    
    # 确保比特长度是bits_per_symbol的整数倍
    if len(bits) % bits_per_symbol != 0:
        padding = bits_per_symbol - (len(bits) % bits_per_symbol)
        bits = np.pad(bits, (0, padding), 'constant')
    
    # 将bits分组
    bits_groups = bits.reshape(-1, bits_per_symbol)
    
    # 转换每组比特为十进制
    decimal_values = np.zeros(len(bits_groups))
    for i in range(bits_per_symbol):
        decimal_values += bits_groups[:, i] * (2 ** (bits_per_symbol - 1 - i))
    
    # 生成Gray码映射的星座点
    # 为简化，我们直接按列优先生成QAM星座点
    symbols = np.zeros(len(decimal_values), dtype=complex)
    
    # 计算实部和虚部的可能值
    amplitude_levels = np.arange(-(K-1), K, 2)
    
    for i, decimal in enumerate(decimal_values):
        # 计算行和列索引
        row = decimal % K
        col = decimal // K
        
        # 计算实部和虚部
        symbols[i] = amplitude_levels[int(col)] + 1j * amplitude_levels[int(row)]
    
    # 归一化能量
    energy = np.mean(np.abs(symbols)**2)
    symbols = symbols / np.sqrt(energy)
    
    return symbols, decimal_values



if __name__ == "__main__":
    # 生成随机二进制比特流
    np.random.seed(0)  # 为了结果可重现
    bits_length = 32
    random_bits = np.random.randint(0, 2, bits_length)
    
    # ===== MPSK调制演示 =====
    # 4-PSK (QPSK) 调制
    M_psk = 4
    complex_symbols_psk, decimal_values_psk = mpsk_modulation(random_bits, M_psk)
    
    # ===== QAM调制演示 =====
    # 16-QAM 调制
    M_qam = 16
    complex_symbols_qam, decimal_values_qam = qam_modulation(random_bits, M_qam)
    