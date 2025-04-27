import numpy as np
from scipy.signal import find_peaks

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

def bpsk_demodulator_with_symbol_sync(fs, f, M, mSig):
    """
    BPSK解调器函数，从BPSK调制的正弦载波中解调数据。
    
    参数:
    fs : float
        采样频率 (Hz)
    f : float
        载波频率 (Hz)
    M : float
        调制指数 M = fc/Rb （每两个周期调制一个比特）
    mSig : np.ndarray
        BPSK调制载波的离散波形数据数组
        
    返回:
    nco_i : np.ndarray
        同相分量波形数据数组
    dmData : np.ndarray
        解调的二进制数据数组
    lpf1 : np.ndarray
        低通滤波器输出
    phase : np.ndarray
        相位数组
    """
    # ==
    # 解调参数
    # ==
    ncoFrequency = f
    ncoInitPhase = -np.pi / 2  # pi / 3
    ncoStep = 5E-5
    lpfDepth = 20
    
    # ==
    # 数据归一化
    # ==
    N = len(mSig)
    T = 1 / f  # 载波周期：1 / 载波频率
    Ts = 1 / fs  # 采样周期：1 / 采样频率
    t = np.arange(0, N) / fs
    
    # ==
    # Costas环路载波恢复和解调
    # ==
    # 初始化处理数组
    carrier = np.zeros(N)
    nco_i = np.zeros(N)
    nco_q = np.zeros(N)
    mix1 = np.zeros(N)
    mix2 = np.zeros(N)
    mix3 = np.zeros(N)
    lpf1 = np.zeros(N)
    lpf2 = np.zeros(N)
    phase = np.zeros(N)
    phase[0] = ncoInitPhase
    
    # 处理数据
    for i in range(N):
        # NCO相位反馈
        if i > 0:
            # 根据反馈调整NCO频率
            phase[i] = phase[i - 1] - (ncoStep * np.pi * np.sign(mix3[i - 1]))
        
        # NCO
        nco_i[i] = np.cos(2 * np.pi * ncoFrequency * t[i] + phase[i])
        nco_q[i] = np.sin(2 * np.pi * ncoFrequency * t[i] + phase[i])
        
        # 输入混频器
        mix1[i] = mSig[i] * nco_i[i]
        mix2[i] = mSig[i] * nco_q[i]
        
        # 低通滤波器
        if i < lpfDepth:
            lpf1[i] = np.sum(mix1[:i+1])
            lpf2[i] = np.sum(mix2[:i+1])
        else:
            lpf1[i] = np.sum(mix1[i-lpfDepth+1:i+1])
            lpf2[i] = np.sum(mix2[i-lpfDepth+1:i+1])
        
        # 反馈混频器
        mix3[i] = lpf1[i] * lpf2[i]
    
    # ==
    # 比特解码
    # ==
    saPerCycl = T / Ts  # 每周期样本数
    saPerSym = saPerCycl * M  # 每符号样本数
    BN = int(np.floor(N / saPerSym))  # 比特数量
    dmData = np.zeros(BN)
    
    # ==
    # 码同步
    # ==
    aa = np.abs(np.diff(lpf1))
    # 在Python中使用scipy.signal的findpeaks函数替代MATLAB的findpeaks
    peakLoc1, _ = find_peaks(aa, prominence=np.sqrt(np.mean(aa**2)))  # MinPeakProminence改为prominence参数
    
    ra = np.mod(peakLoc1, saPerSym)
    # 使用np.bincount找出最常见的值（替代MATLAB的mode函数）
    counts = np.bincount(ra.astype(int))
    raloc = np.argmax(counts)
    
    for i in range(BN):
        tt = int(i * saPerSym - 0.4 * saPerSym + raloc)
        
        if i == BN - 1:  # Python索引从0开始，所以这里是BN-1
            end_idx = min(tt + int(0.4 * saPerSym - raloc), N)
            PG1 = np.mean(lpf1[tt:end_idx])
        else:
            end_idx = min(tt + int(0.5 * saPerSym), N)
            PG1 = np.mean(lpf1[tt:end_idx])
        
        if PG1 >= 0:
            dmData[i] = 1
        else:
            dmData[i] = 0
    
    return nco_i, dmData, lpf1, phase, raloc


def Error110Func(rxData):
    """
    Calculate error rate based on comparison with known pattern.
    先计算相关性对齐再计算误码率。
    
    Parameters:
    rxData : np.ndarray
        Received data bits
        
    Returns:
    float
        Error rate
    """
    
    # 定义期望的模式
    pattern = np.array([1, 1, 0])
    
    # 创建完整的参考序列
    expected_pattern_full = np.tile(pattern, len(rxData)//3 + 1)[:len(rxData)]
    
    # 计算不同偏移量下的相关性
    max_offset = len(pattern)
    best_correlation = -1
    best_offset = 0
    
    for offset in range(max_offset):
        # 创建当前偏移下的参考序列
        shifted_pattern = np.roll(expected_pattern_full, offset)
        
        # 计算相关性 (匹配的位数)
        correlation = np.sum(rxData == shifted_pattern)
        
        # 更新最佳偏移
        if correlation > best_correlation:
            best_correlation = correlation
            best_offset = offset
    
    # 使用最佳偏移量创建对齐后的参考序列
    aligned_pattern = np.roll(expected_pattern_full, best_offset)
    
    # 计算误码率
    errors = np.sum(rxData != aligned_pattern)
    error_rate = errors / len(rxData)
    
    return aligned_pattern, error_rate