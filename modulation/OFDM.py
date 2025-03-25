import numpy as np

def ofdm_modulation(modulated_symbols, n_fft, n_cp, modulation_type=None, pilot_pattern='edge', comb_num=8):
    """
    通用OFDM调制函数，支持不同的调制方式
    
    参数:
    modulated_symbols: 已调制的符号序列 (如QPSK, 16QAM, 8PSK等)
    n_fft: FFT大小
    n_cp: 循环前缀长度
    modulation_type: 调制方式描述 (可选，用于记录)
    pilot_pattern: 导频模式 ('edge': 边缘导频, 'comb': 梳状导频, 'scattered': 分散导频)
    
    返回:
    ofdm_signal: OFDM时域信号
    frame_structure: OFDM帧结构信息，包含每个OFDM符号中的导频和数据位置
    """
    # 获取符号数量
    num_symbols = len(modulated_symbols)
    
    # 设置导频配置
    if pilot_pattern == 'edge':
        # 边缘导频模式: 在子载波边缘放置导频
        pilot_indices = [1, n_fft//2-1]
        pilot_values = np.array([1+0j, 1+0j])
        num_pilots = len(pilot_indices)
    elif pilot_pattern == 'comb':
        # 梳状导频模式: 每隔几个子载波放置一个导频
        pilot_spacing = comb_num  # 每8个子载波放一个导频
        pilot_indices = [i for i in range(1, n_fft//2, pilot_spacing)]
        pilot_values = np.array([1+0j] * len(pilot_indices))
        num_pilots = len(pilot_indices)
    elif pilot_pattern == 'scattered':
        # 分散导频模式: 在时频网格上分散放置导频
        # 这种情况下，导频位置会随OFDM符号索引变化
        base_indices = [1, n_fft//8, n_fft//4, 3*n_fft//8, n_fft//2-1]
        pilot_values = np.array([1+0j] * len(base_indices))
        num_pilots = len(base_indices)
    else:
        # 默认使用边缘导频
        pilot_indices = [1, n_fft//2-1]
        pilot_values = np.array([1+0j, 1+0j])
        num_pilots = len(pilot_indices)
    
    # 计算每个OFDM符号可以携带的数据符号数（考虑DC子载波）
    dc_null = 1  # DC子载波通常置零
    nyquist_null = 1  # Nyquist频率子载波也置零（当n_fft为偶数时）
      
    data_carriers_per_symbol = n_fft//2 - num_pilots - dc_null - nyquist_null
    
    # 计算需要的OFDM符号数
    num_ofdm_symbols = int(np.ceil(num_symbols / data_carriers_per_symbol))
    
    # 创建OFDM调制的频域数据
    ofdm_freq_data = np.zeros((num_ofdm_symbols, n_fft), dtype=complex)
    
    # 记录帧结构信息
    frame_structure = {
        'num_ofdm_symbols': num_ofdm_symbols,
        'n_fft': n_fft,
        'n_cp': n_cp,
        'modulation_type': modulation_type,
        'pilot_pattern': pilot_pattern,
        'symbol_mapping': []  # 用于存储每个OFDM符号的子载波映射关系
    }
    
    # 将调制符号放入OFDM频域数据
    for i in range(num_ofdm_symbols):
        # 创建当前OFDM符号的子载波映射
        subcarrier_map = np.zeros(n_fft, dtype=int) 
        # -1: DC空/Nyquist空, 0: 未使用, 1: 数据, 2: 导频, 3: 厄密特对称
        subcarrier_map[0] = -1  # DC子载波
        subcarrier_map[n_fft//2] = -1  # Nyquist频率子载波
        
        # 设置导频
        if pilot_pattern == 'scattered':
            # 分散导频: 基于OFDM符号索引计算导频位置
            pilot_indices = [(idx + i) % (n_fft//2) for idx in base_indices]
            pilot_indices = [idx for idx in pilot_indices if idx != 0 and idx != n_fft//2]  # 避开DC和Nyquist子载波
        
        for j, pilot_idx in enumerate(pilot_indices):
            ofdm_freq_data[i, pilot_idx] = pilot_values[j % len(pilot_values)]
            ofdm_freq_data[i, n_fft-pilot_idx] = np.conjugate(pilot_values[j % len(pilot_values)])
            subcarrier_map[pilot_idx] = 2
            subcarrier_map[n_fft-pilot_idx] = 3 
        
        # 计算当前OFDM符号的数据起始和结束索引
        start_idx = i * data_carriers_per_symbol
        end_idx = min(start_idx + data_carriers_per_symbol, num_symbols)
        
        # 实际使用的数据符号数
        actual_data_len = end_idx - start_idx
        
        # 将调制符号放入频域数据（跳过DC子载波、Nyquist子载波和导频）
        data_idx = 0
        for j in range(1, n_fft//2):
            if j in pilot_indices or j == n_fft//2:
                continue  # 跳过导频位置和Nyquist频率
            if data_idx < actual_data_len:
                ofdm_freq_data[i, j] = modulated_symbols[start_idx + data_idx]
                ofdm_freq_data[i, n_fft-j] = np.conjugate(modulated_symbols[start_idx + data_idx])
                subcarrier_map[j] = 1  # 标记为数据子载波
                subcarrier_map[n_fft-j] = 3  # 标记为厄密特对称的数据子载波
                data_idx += 1
            else:
                break
        
        # 记录子载波映射
        frame_structure['symbol_mapping'].append(subcarrier_map.tolist())
    
    # 执行IFFT获取时域OFDM符号
    ofdm_time_symbols = np.fft.ifft(ofdm_freq_data, axis=1)
    
    # 添加循环前缀
    ofdm_time_symbols_cp = np.zeros((num_ofdm_symbols, n_fft + n_cp), dtype=complex)
    for i in range(num_ofdm_symbols):
        # 添加CP
        ofdm_time_symbols_cp[i, :n_cp] = ofdm_time_symbols[i, -n_cp:]
        # 添加OFDM符号
        ofdm_time_symbols_cp[i, n_cp:] = ofdm_time_symbols[i]
    
    # 将所有OFDM符号串联成一个序列
    ofdm_signal = ofdm_time_symbols_cp.flatten()
    
    return ofdm_signal, frame_structure
