import numpy as np


def mpsk_demodulation(complex_symbols, M):
    """
    实现MPSK解调，TODO：加入相位同步和噪声处理
    
    参数:
    complex_symbols -- 接收到的复数符号点
    M -- 相位数，必须是2的幂(如2, 4, 8, 16等)
    
    返回:
    bits -- 解调后的二进制比特流
    decoded_symbols -- 解调的符号索引
    """
    if not (M & (M-1) == 0) or M < 2:
        raise ValueError("M必须是2的幂，如2, 4, 8, 16等")
    
    # 每个符号的比特数
    bits_per_symbol = int(np.log2(M))
    
    # 计算每个符号点的相位角度（弧度）
    received_phases = np.angle(complex_symbols)
    
    # 将负相位角转换为正相位角（0到2π范围内）
    received_phases = np.mod(received_phases, 2 * np.pi)
    
    # 将接收到的相位映射到最近的理想相位点
    # 理想相位点间隔为2π/M
    decoded_symbols = np.round(received_phases / (2 * np.pi / M)) % M
    
    # 将解调的符号转换为比特序列
    bits = np.zeros(len(decoded_symbols) * bits_per_symbol, dtype=int)
    
    for i in range(len(decoded_symbols)):
        # 将每个符号转换为二进制比特序列
        symbol_bits = format(int(decoded_symbols[i]), f'0{bits_per_symbol}b')
        start_idx = i * bits_per_symbol
        for j in range(bits_per_symbol):
            bits[start_idx + j] = int(symbol_bits[j])
    
    return bits, decoded_symbols

def qam_demodulation(complex_symbols, M):
    """
    实现QAM(Quadrature Amplitude Modulation)解调
    
    参数:
    complex_symbols -- 接收到的复数符号点
    M -- 调制阶数，必须是平方数(如4, 16, 64, 256等)
    
    返回:
    bits -- 解调后的二进制比特流
    decimal_values -- 解调后的十进制值
    """
    # 检查M是否为平方数
    K = int(np.sqrt(M))
    if K * K != M:
        raise ValueError("M必须是平方数，如4, 16, 64, 256等")
    
    # 每个符号的比特数
    bits_per_symbol = int(np.log2(M))
    
    # 首先对接收到的符号进行能量归一化
    # 估计接收符号的平均能量
    rx_energy = np.mean(np.abs(complex_symbols)**2)
    # 归一化接收符号
    normalized_symbols = complex_symbols / np.sqrt(rx_energy)
    
    # 生成理想的QAM星座点（与调制时相同的方式）
    amplitude_levels = np.arange(-(K-1), K, 2)
    ideal_symbols = np.zeros((M,), dtype=complex)
    decimal_mapping = np.zeros((M,), dtype=int)
    
    idx = 0
    for col in range(K):
        for row in range(K):
            # 生成星座点
            ideal_symbols[idx] = amplitude_levels[col] + 1j * amplitude_levels[row]
            # 记录对应的十进制值
            decimal_mapping[idx] = row * K + col
            idx += 1
    
    # 归一化理想星座点的能量
    ideal_energy = np.mean(np.abs(ideal_symbols)**2)
    ideal_symbols = ideal_symbols / np.sqrt(ideal_energy)
    
    # 解调：为每个接收符号找到最近的理想星座点
    decimal_values = np.zeros(len(normalized_symbols), dtype=int)
    
    for i, symbol in enumerate(normalized_symbols):
        # 计算接收符号与所有理想星座点的欧氏距离
        distances = np.abs(ideal_symbols - symbol)
        # 找出距离最小的星座点索引
        min_idx = np.argmin(distances)
        # 映射回十进制值
        decimal_values[i] = decimal_mapping[min_idx]
    
    # 将十进制值转换为比特序列
    bits = np.zeros(len(decimal_values) * bits_per_symbol, dtype=int)
    
    for i, decimal in enumerate(decimal_values):
        # 将十进制值转换为二进制比特
        bit_sequence = np.binary_repr(decimal, width=bits_per_symbol)
        # 填充到结果数组
        for j, bit in enumerate(bit_sequence):
            bits[i * bits_per_symbol + j] = int(bit)
    
    return bits, decimal_values