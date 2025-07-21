import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import os
from .Analyze import range_fft
from .ReadData import read_dca1000, radar_params_extract
import glob
from scipy.ndimage import median_filter

def find_max_energy_range_bin(data_range_fft, channel_num=0):
   """
   在距离维度上找到能量最大的单元。
   
   参数:
   data_range_fft : range_fft结果，可以是以下维度:
                    - (num_samples, num_total_chirps)
                    - (num_channels, num_samples, num_total_chirps)
   channel_num : int, 可选
                 当data_range_fft为3维时，指定要处理的通道索引
                 默认为0
   
   返回值:
   tuple: (range_bin_data, max_range_idx)
          range_bin_data : 能量最大的距离单元的数据
          max_range_idx : 能量最大的距离单元索引
   """
   # 根据输入数据的维度进行处理
   if data_range_fft.ndim == 2:
       # 2D情况: (num_samples, num_total_chirps)
       abs_sum = np.sum(np.abs(data_range_fft), axis=1)
       max_range_idx = np.argmax(abs_sum)
       range_bin_data = data_range_fft[max_range_idx, :]
   
   elif data_range_fft.ndim == 3:
       # 3D情况: (num_channels, num_samples, num_total_chirps)
       abs_sum = np.sum(np.abs(data_range_fft[channel_num, :, :]), axis=1)
       max_range_idx = np.argmax(abs_sum)
       range_bin_data = data_range_fft[channel_num, max_range_idx, :]
   
   else:
       raise ValueError("输入数据维度必须是2或3")
   
   return range_bin_data, max_range_idx


def extract_and_unwrap_phase(range_bin_data, maxloc=None, method='adaptive', 
                           filter_noise=True, distance_resolu=0.0375, f=77e9):
    """
    提取相位并进行改进的解缠绕处理，适应距离单元跳变。
    
    参数:
    range_bin_data : 单一距离单元的复数数据 或 2D数组 (time, range)
    maxloc : 每个时刻的最大能量距离单元索引（可选）
    method : 解缠绕方法 ('adaptive', 'standard', 'compensated')
    filter_noise : 是否预先滤波
    distance_resolu : 距离分辨率 (m)
    f : 雷达频率 (Hz)
    
    返回值:
    unwrapped_phase : 解缠绕后的相位时间序列
    magnitude : 信号幅度
    """
    
    c = 3e8
    wavelength = c / f
    
    # 处理输入数据
    if range_bin_data.ndim == 1:
        # 单一距离单元数据
        phase = np.angle(range_bin_data)
        magnitude = np.abs(range_bin_data)
        
        if filter_noise:
            phase = median_filter(phase, size=3)
        
        if method == 'standard':
            unwrapped_phase = np.unwrap(phase)
        else:
            unwrapped_phase = adaptive_unwrap_1d(phase, magnitude)
        
        return unwrapped_phase, magnitude
    
    # 2D数据：需要跟踪距离单元
    time_steps, num_ranges = range_bin_data.shape
    
    if method == 'compensated' and maxloc is not None:
        # 使用距离单元跳变补偿的方法
        return unwrap_with_range_compensation(range_bin_data, maxloc, 
                                            filter_noise, distance_resolu, wavelength)
    else:
        # 自动跟踪最大能量距离单元
        return unwrap_with_tracking(range_bin_data, filter_noise)

def unwrap_with_range_compensation(range_bin_data, maxloc, filter_noise, distance_resolu, wavelength):
    """
    使用距离单元跳变补偿的解缠绕方法
    """
    time_steps = range_bin_data.shape[0]
    
    # 提取每个时刻对应距离单元的相位和幅度
    phases = []
    magnitudes = []
    range_indices = []
    
    for t in range(time_steps):
        if t < len(maxloc):
            range_idx = int(maxloc[t])
            # 确保索引在有效范围内
            if 0 <= range_idx < range_bin_data.shape[1]:
                complex_val = range_bin_data[t, range_idx]
                phases.append(np.angle(complex_val))
                magnitudes.append(np.abs(complex_val))
                range_indices.append(range_idx)
            else:
                # 如果索引超出范围，使用最大能量距离单元
                power_slice = np.abs(range_bin_data[t, :])**2
                best_idx = np.argmax(power_slice)
                complex_val = range_bin_data[t, best_idx]
                phases.append(np.angle(complex_val))
                magnitudes.append(np.abs(complex_val))
                range_indices.append(best_idx)
        else:
            # 如果maxloc长度不够，自动寻找最大能量
            power_slice = np.abs(range_bin_data[t, :])**2
            best_idx = np.argmax(power_slice)
            complex_val = range_bin_data[t, best_idx]
            phases.append(np.angle(complex_val))
            magnitudes.append(np.abs(complex_val))
            range_indices.append(best_idx)
    
    phases = np.array(phases)
    magnitudes = np.array(magnitudes)
    range_indices = np.array(range_indices)
    
    # 检测并补偿距离单元跳变
    compensated_phases = compensate_range_jumps(phases, range_indices, distance_resolu, wavelength)
    
    # 噪声滤波
    if filter_noise:
        from scipy.ndimage import median_filter
        compensated_phases = median_filter(compensated_phases, size=3)
    
    # 自适应解缠绕
    unwrapped_phase = adaptive_unwrap_1d(compensated_phases, magnitudes)
    
    print(f"距离单元跳变补偿完成，跳变次数: {np.sum(np.diff(range_indices) != 0)}")
    
    return unwrapped_phase, magnitudes

def unwrap_with_tracking(range_bin_data, filter_noise, search_range=10):
    """
    自动跟踪最大能量距离单元的解缠绕方法
    """
    time_steps, num_ranges = range_bin_data.shape
    
    # 寻找初始最大能量距离单元
    initial_powers = np.mean(np.abs(range_bin_data)**2, axis=0)
    current_range = np.argmax(initial_powers)
    
    phases = []
    magnitudes = []
    range_indices = []
    
    for t in range(time_steps):
        # 在当前位置附近搜索最大能量
        search_start = max(0, current_range - search_range)
        search_end = min(num_ranges, current_range + search_range + 1)
        
        range_slice = range_bin_data[t, search_start:search_end]
        power_slice = np.abs(range_slice)**2
        local_max_idx = np.argmax(power_slice)
        actual_range_idx = search_start + local_max_idx
        
        complex_val = range_bin_data[t, actual_range_idx]
        phases.append(np.angle(complex_val))
        magnitudes.append(np.abs(complex_val))
        range_indices.append(actual_range_idx)
        
        # 更新搜索中心
        current_range = actual_range_idx
    
    phases = np.array(phases)
    magnitudes = np.array(magnitudes)
    
    # 噪声滤波
    if filter_noise:
        print("应用噪声滤波...")
        from scipy.ndimage import median_filter
        phases = median_filter(phases, size=3)
    
    # 自适应解缠绕
    unwrapped_phase = adaptive_unwrap_1d(phases, magnitudes)
    
    return unwrapped_phase, magnitudes

def compensate_range_jumps(phases, range_indices, distance_resolu, wavelength):
    """
    补偿距离单元跳变引起的相位偏移
    """
    compensated_phases = phases.copy()
    range_jumps = np.diff(range_indices)
    jump_positions = np.where(range_jumps != 0)[0]
    
    if len(jump_positions) == 0:
        return compensated_phases
    
    print(f"检测到 {len(jump_positions)} 个距离单元跳变，进行相位补偿...")
    
    cumulative_compensation = 0
    
    for jump_pos in jump_positions:
        # 计算距离变化引起的相位变化
        range_change = range_jumps[jump_pos]
        distance_change = range_change * distance_resolu
        phase_compensation = 4 * np.pi * distance_change / wavelength
        
        # 累积补偿（影响后续所有点）
        cumulative_compensation += phase_compensation
        compensated_phases[jump_pos + 1:] -= cumulative_compensation
        
        if jump_pos < 10:  # 只打印前10个跳变信息
            print(f"  时刻 {jump_pos+1}: 距离单元跳变 {range_change}, 补偿相位 {phase_compensation:.3f} rad")
    
    return compensated_phases

def adaptive_unwrap_1d(phases, magnitudes, confidence_threshold=0.3):
    """
    基于信号质量的1D自适应相位解缠绕
    """
    unwrapped = np.zeros_like(phases)
    unwrapped[0] = phases[0]
    
    # 计算信号质量权重
    normalized_mag = magnitudes / np.max(magnitudes)
    quality_weights = np.where(normalized_mag > confidence_threshold, 1.0, normalized_mag)
    
    for i in range(1, len(phases)):
        # 计算相位差
        phase_diff = phases[i] - unwrapped[i-1]
        
        # 标准解缠绕
        while phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        while phase_diff < -np.pi:
            phase_diff += 2 * np.pi
        
        # 对于低质量信号，检查是否需要额外处理
        if quality_weights[i] < confidence_threshold and abs(phase_diff) > np.pi * 0.8:
            # 使用趋势预测
            if i >= 3:
                recent_diffs = np.diff(unwrapped[i-3:i])
                if len(recent_diffs) > 0:
                    trend = np.median(recent_diffs)  # 使用中位数更鲁棒
                    predicted_phase = unwrapped[i-1] + trend
                    
                    # 重新计算相位差
                    alt_phase_diff = phases[i] - predicted_phase
                    while alt_phase_diff > np.pi:
                        alt_phase_diff -= 2 * np.pi
                    while alt_phase_diff < -np.pi:
                        alt_phase_diff += 2 * np.pi
                    
                    # 如果预测的相位差更小，使用它
                    if abs(alt_phase_diff) < abs(phase_diff):
                        unwrapped[i] = predicted_phase + alt_phase_diff
                        continue
        
        unwrapped[i] = unwrapped[i-1] + phase_diff
    
    return unwrapped

def process_micro_phase(unwrapped_phase, times, times_compen, window_size=51, poly_order=2, threshold=0.02):
    """
    处理相位数据以提取微信息。
    参数:
    unwrapped_phase : np.ndarray
        原始相位数据，可以是复数形式
    window_size : int, 可选
        Savitzky-Golay滤波器的窗口大小，应为奇数，默认为51
    poly_order : int, 可选
        Savitzky-Golay滤波器的多项式阶数，默认为2
    threshold : float, 可选
        相位阈值，用于抑制异常值，默认为0.02
        
    返回值:
    tuple: (processed_phase, smooth_phase, micro_doppler_phase)
        processed_phase : 阈值处理后的微多相位
        smooth_phase : 平滑后的相位趋势
        micro_doppler_phase : 未阈值处理的微多相位
    """
    
    spline_interpolator = interp1d(times, unwrapped_phase, kind='cubic', bounds_error=False, fill_value="extrapolate")
    unwrapped_phase = spline_interpolator(times_compen)
   
    # 确保滤波窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    # 应用Savitzky-Golay滤波
    smooth_phase = savgol_filter(unwrapped_phase, window_size, poly_order)
    
    # 提取微相位变化
    micro_phase = unwrapped_phase - smooth_phase
    
    # 阈值处理
    processed_phase = micro_phase.copy()
    processed_phase[processed_phase > threshold] = 1e-3
    processed_phase[processed_phase < -threshold] = -1e-3
    
    processed_phase = uniform_filter1d(processed_phase, size=3)
    
    return processed_phase, smooth_phase, micro_phase

def create_time_arrays(ChirpPeriod, FrameNum, fullChirp):
    """
    创建用于雷达信号处理的时间数组

    参数:
    ChirpPeriod : float
        Chirp周期，单位为μs
    FrameNum : list or array
        帧索引数组
    fullChirp : float
        完整帧周期内的chirp数量 (FramPeriod/ChirpPeriod)
        
    返回值:
    tuple
        (times, times_compen) - 标准时间数组和补偿时间数组，单位为ms
    """
    # 创建帧索引数组，并调整为0索引
    frame_indices = np.array(FrameNum) - 1

    # 创建标准chirp索引数组 (1-255)
    chirp_indices = np.arange(1, 256).reshape(-1, 1)

    # 计算标准时间点，单位为ms
    times = ChirpPeriod * (frame_indices * fullChirp + chirp_indices) / 1000

    # 创建完整chirp索引数组 (1-fullChirp)
    chirp_indices_compen = np.arange(1, fullChirp + 1).reshape(-1, 1)

    # 计算补偿时间点，单位为ms
    times_compen = ChirpPeriod * (frame_indices * fullChirp + chirp_indices_compen) / 1000

    # 转置并展平数组
    times = times.T.flatten()
    times_compen = times_compen.T.flatten()

    return times, times_compen

def extract_phase_from_max_range_bin(data_range_fft, max_range_idx, range_search=3, channel_num=0, time_increment=1):
    """
    从最大能量距离单元附近提取相位信息

    参数:
    data_range_fft : np.ndarray
        Range-FFT结果，维度为(num_channels, num_samples, num_time_samples)
    max_range_idx : int
        总体最大能量的距离单元索引
    range_search : int, 可选
        在max_range_idx周围搜索的距离单元范围，默认为3
    channel_num : int, 可选
        要处理的通道索引，默认为0
    time_increment : int, 可选
        时间采样的增量，默认为1
        
    返回值:
    tuple: (phase_range, max_locations)
        phase_range : np.ndarray
            提取的相位信息数组
        max_locations : list
            每个时间点的局部最大能量位置
    """
    if data_range_fft.ndim == 2:
        # 2D情况: (num_samples, num_time_samples)
        num_samples, num_time_samples = data_range_fft.shape
        use_channel = False
    elif data_range_fft.ndim == 3:
        # 3D情况: (num_channels, num_samples, num_time_samples)
        num_channels, num_samples, num_time_samples = data_range_fft.shape
        use_channel = True
        if channel_num >= num_channels:
            raise ValueError(f"channel_num ({channel_num}) 超出可用通道数 ({num_channels})")
    else:
        raise ValueError(f"不支持的数据维度: {data_range_fft.ndim}D，期望2D或3D")


    maxloc = []
    phase_range = []

    # 处理每个时间采样点
    for tn in range(0, num_time_samples, time_increment):
        # 根据数据维度选择数据切片方式
        if use_channel:
            # 3D数据：使用指定通道
            range_slice = np.abs(data_range_fft[channel_num, 
                                             max_range_idx-range_search:max_range_idx+range_search+1, 
                                             tn])
        else:
            # 2D数据：直接使用
            range_slice = np.abs(data_range_fft[max_range_idx-range_search:max_range_idx+range_search+1, 
                                              tn])
        
        maxV = np.max(range_slice)
        maxloc_tn = np.argmax(range_slice)
        
        # 处理零值情况，使用前一个位置
        if maxV == 0 and tn > 0:
            maxloc_tn = maxloc[-1]
            
        maxloc.append(maxloc_tn)
        
        # 计算相位索引并提取相位信息
        phase_idx = maxloc_tn + max_range_idx - range_search
        
        if use_channel:
            # 3D数据
            phase_range.append(data_range_fft[channel_num, phase_idx, tn])
        else:
            # 2D数据
            phase_range.append(data_range_fft[phase_idx, tn])
    
    # 转换为numpy数组
    phase_range = np.array(phase_range)
    maxloc_fix = maxloc + max_range_idx - range_search
    
    return phase_range, maxloc_fix

def extract_processed_radar_phase(file_name):
    fc = 200
    FrameNum = list(range(1, 257)) 
    lamda = 3e8 / 77e9
    ChannlNum = 0
    Rb = 100
    fc = 200
    modulationIndex = fc / Rb  # Modulate at one bit per two cycles

    # Load rawData3D
    folder = 'data/exp'  # 指定包含.bin文件的文件夹路径

    file_path = os.path.join(folder, file_name)

    file_name = os.path.basename(file_path)
    rawData = read_dca1000(file_path)

    params = radar_params_extract(file_path)
    ADCSample, ChirpPeriod, ADCFs, ChirpNum, FramPeriod, FramNum, slope, BandWidth, R_Maximum, R_resulo, V_Maximum, V_resulo = params
    fs = 1e6 / ChirpPeriod

    Len = rawData.shape[1]
    fullChirp = FramPeriod / ChirpPeriod

    times, times_compen = create_time_arrays(ChirpPeriod, FrameNum, fullChirp)

    processed_phases = []
    for ChannlNum in range(4):
        frames_dimension = int(round(Len/(ADCSample*ChirpNum)))
        Data_all = np.reshape(rawData, (4, int(ADCSample), int(ChirpNum), frames_dimension), order='F')
        proData = np.reshape(Data_all[:, :, :, np.array(FrameNum)-1], (4, int(ADCSample), -1), order='F')

        DataRangeFft, _ = range_fft(proData, int(ADCSample), BandWidth, apply_window=False)
        _, maxlocAll = find_max_energy_range_bin(DataRangeFft[ChannlNum, :, :])
        phase_range = extract_phase_from_max_range_bin(DataRangeFft, maxlocAll, range_search=3, channel_num=ChannlNum, time_increment=1)

        # Process max power range bin 
        unwrapped_phase, _ = extract_and_unwrap_phase(phase_range)
        processed_phase, _, _ = process_micro_phase(unwrapped_phase, times, times_compen, window_size=57, poly_order=3, threshold=0.02)
        
        processed_phases.append(processed_phase)
    
    return processed_phases, fs, fc, modulationIndex