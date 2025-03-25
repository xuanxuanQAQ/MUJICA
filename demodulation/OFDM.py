import numpy as np
from scipy import interpolate

def ofdm_demodulation(ofdm_signal, frame_structure, equalization='zf'):
    """
    完整的OFDM解调函数，组合了上述三个函数
    
    参数:
    ofdm_signal: 接收到的OFDM时域信号
    frame_structure: OFDM帧结构信息（由ofdm_modulation返回）
    equalization: 均衡方法 ('zf': 零强制均衡, 'mmse': 最小均方误差均衡)
    
    返回:
    demodulated_symbols: 解调后的符号序列
    channel_estimates: 信道估计结果
    """
    # 1. 预处理 - 移除CP并进行FFT
    ofdm_freq_symbols, symbol_mapping = ofdm_preprocessing(ofdm_signal, frame_structure)
    
    # 2. 信道估计
    n_fft = frame_structure['n_fft']
    channel_estimates = estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)
    
    # 3. 均衡和解调
    demodulated_symbols = apply_equalization(ofdm_freq_symbols, channel_estimates, symbol_mapping, equalization)
    
    return demodulated_symbols, channel_estimates

def ofdm_preprocessing(ofdm_signal, frame_structure):
    """
    OFDM信号预处理（解调前的处理）
    
    参数:
    ofdm_signal: 接收到的OFDM时域信号
    frame_structure: OFDM帧结构信息（由ofdm_modulation返回）
    
    返回:
    ofdm_freq_symbols: FFT后的频域OFDM符号
    symbol_mapping: 子载波映射信息
    """
    # 从帧结构中提取参数
    num_ofdm_symbols = frame_structure['num_ofdm_symbols']
    n_fft = frame_structure['n_fft']
    n_cp = frame_structure['n_cp']
    symbol_mapping = frame_structure['symbol_mapping']
    
    # 计算每个OFDM符号的总长度
    ofdm_symbol_len = n_fft + n_cp
    
    # 提取OFDM符号（移除循环前缀）
    ofdm_time_symbols = np.zeros((num_ofdm_symbols, n_fft), dtype=complex)
    for i in range(num_ofdm_symbols):
        start_idx = i * ofdm_symbol_len + n_cp  # 跳过CP
        end_idx = (i + 1) * ofdm_symbol_len
        ofdm_time_symbols[i] = ofdm_signal[start_idx:end_idx]
    
    # 执行FFT获取频域OFDM符号
    ofdm_freq_symbols = np.fft.fft(ofdm_time_symbols, axis=1)
    
    return ofdm_freq_symbols, symbol_mapping

def estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft):
    """
    OFDM信道估计
    
    参数:
    ofdm_freq_symbols: FFT后的频域OFDM符号
    symbol_mapping: 子载波映射信息
    n_fft: FFT点数
    
    返回:
    channel_estimates: 信道估计结果
    """
    num_ofdm_symbols = ofdm_freq_symbols.shape[0]
    
    # 存储信道估计结果
    channel_estimates = np.zeros((num_ofdm_symbols, n_fft), dtype=complex)
    
    # 处理每个OFDM符号
    for i in range(num_ofdm_symbols):
        # 获取当前OFDM符号的子载波映射
        current_mapping = np.array(symbol_mapping[i])
        
        # 找出导频位置
        pilot_indices = np.where(current_mapping == 2)[0]
        
        # 导频的理想值（发送端设置的导频值）
        pilot_values = np.array([1+0j] * len(pilot_indices))
        
        # 从接收符号中提取导频
        received_pilots = ofdm_freq_symbols[i, pilot_indices]
        
        # 在导频位置估计信道
        h_pilots = received_pilots / pilot_values
        
        # 使用插值方法估计所有子载波上的信道响应
        if len(pilot_indices) >= 2:  # 至少需要两个导频点才能进行插值
            # 创建插值函数
            h_interpolator_real = interpolate.interp1d(
                pilot_indices, 
                np.real(h_pilots), 
                kind='linear', 
                bounds_error=False,
                fill_value=(np.real(h_pilots[0]), np.real(h_pilots[-1]))
            )
            
            h_interpolator_imag = interpolate.interp1d(
                pilot_indices, 
                np.imag(h_pilots), 
                kind='linear', 
                bounds_error=False,
                fill_value=(np.imag(h_pilots[0]), np.imag(h_pilots[-1]))
            )
            
            # 对前半部分子载波进行信道插值
            front_indices = np.arange(n_fft//2 + 1)  # 包括DC和Nyquist频率
            h_front = h_interpolator_real(front_indices) + 1j * h_interpolator_imag(front_indices)
            
            # 对后半部分使用厄密特对称填充
            h_back = np.conjugate(np.flip(h_front[1:n_fft//2]))  # 排除DC和Nyquist
            
            # 组合完整的信道估计
            h_interpolated = np.concatenate([h_front, h_back])
            
            # 存储信道估计结果
            channel_estimates[i] = h_interpolated
        else:
            # 如果导频不足，则假设平坦信道
            # 注意：即使是平坦信道，也要维持厄密特对称性
            h_front = np.mean(h_pilots) * np.ones(n_fft//2 + 1)
            h_back = np.conjugate(np.flip(h_front[1:n_fft//2]))  # 排除DC和Nyquist
            channel_estimates[i] = np.concatenate([h_front, h_back])
    
    return channel_estimates

def apply_equalization(ofdm_freq_symbols, channel_estimates, symbol_mapping, equalization='zf'):
    """
    应用信道均衡并提取数据符号
    
    参数:
    ofdm_freq_symbols: FFT后的频域OFDM符号
    channel_estimates: 信道估计结果
    symbol_mapping: 子载波映射信息
    equalization: 均衡方法 ('zf': 零强制均衡, 'mmse': 最小均方误差均衡)
    
    返回:
    demodulated_symbols: 解调后的符号序列
    """
    num_ofdm_symbols = ofdm_freq_symbols.shape[0]
    
    # 存储解调后的符号
    demodulated_symbols = []
    
    # 处理每个OFDM符号
    for i in range(num_ofdm_symbols):
        # 获取当前OFDM符号的子载波映射
        current_mapping = np.array(symbol_mapping[i])
        
        # 应用信道均衡
        if equalization == 'zf':  # 零强制均衡
            equalized_symbols = ofdm_freq_symbols[i] / channel_estimates[i]
        elif equalization == 'mmse':  # 最小均方误差均衡
            # 假设噪声方差为0.1（实际应用中应该估计）
            noise_var = 0.1
            equalized_symbols = ofdm_freq_symbols[i] * np.conj(channel_estimates[i]) / (np.abs(channel_estimates[i])**2 + noise_var)
        else:  # 默认使用零强制均衡
            equalized_symbols = ofdm_freq_symbols[i] / channel_estimates[i]
        
        # 找出数据子载波位置
        data_indices = np.where(current_mapping == 1)[0]
        
        # 提取数据符号
        data_symbols = equalized_symbols[data_indices]
        
        # 添加到解调符号列表
        demodulated_symbols.extend(data_symbols)
    
    # 转换为numpy数组
    demodulated_symbols = np.array(demodulated_symbols)
    
    return demodulated_symbols