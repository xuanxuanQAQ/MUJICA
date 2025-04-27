import numpy as np

def generate_random_binary(length):
    """
    生成指定长度的随机二进制序列
    
    参数:
    length (int): 要生成的二进制位数
    
    返回:
    numpy.ndarray: 包含0和1的随机二进制序列
    """
    # 使用numpy生成随机二进制序列
    binary_sequence = np.random.randint(0, 2, length)
    
    return binary_sequence

def generate_bpsk_signal(fs, fc, M, bit_pattern=[1, 1, 0], num_repetitions=1):
    """
    生成BPSK调制信号
    
    参数:
    fs : float
        采样频率(Hz)
    fc : float
        载波频率(Hz)
    M : float
        调制指数(每个比特的载波周期数)
    bit_pattern : list
        要重复的比特模式(默认为[1,1,0])
    num_repetitions : int
        重复bit_pattern的次数
        
    返回值:
    np.ndarray
        BPSK调制信号
    list
        原始比特序列
    """
    # 重复比特模式以创建完整序列
    bits = bit_pattern * num_repetitions
    
    # 计算每个符号的样本数
    samples_per_symbol = int(fs / (fc / M))
    
    # 生成时间向量
    total_samples = len(bits) * samples_per_symbol
    t = np.arange(total_samples) / fs
    
    # 生成载波
    carrier = np.cos(2 * np.pi * fc * t)
    
    # 生成BPSK信号
    bpsk_signal = np.zeros_like(t)
    
    for i, bit in enumerate(bits):
        start_idx = i * samples_per_symbol
        end_idx = (i + 1) * samples_per_symbol
        
        # 根据比特值调制载波相位(0°或180°)
        phase = 0 if bit == 1 else np.pi
        bpsk_signal[start_idx:end_idx] = np.cos(2 * np.pi * fc * t[start_idx:end_idx] + phase)
    
    return bpsk_signal, bits

