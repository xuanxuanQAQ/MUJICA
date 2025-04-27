import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

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


def extract_and_unwrap_phase(range_bin_data):
    """
    提取相位并进行解缠绕处理。
    
    参数:
    range_bin_data : 单一距离单元的复数数据
    
    返回值:
    unwrapped_phase : 解缠绕后的相位时间序列
    magnitude : 信号幅度
    """
    phase = np.angle(range_bin_data)
    magnitude = np.abs(range_bin_data)
    unwrapped_phase = np.unwrap(phase)
    
    return unwrapped_phase, magnitude


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


def process_micromotion_phase(data_range_fft):
    """
    处理雷达距离-多普勒图以提取和分析微动分析的相位信息。
    
    参数: 
    data_range_fft : range_fft结果，维度为(num_samples, num_total_chirps)
    
    返回值: 
    processed_phase : 去除异常值后的阈值化微多普勒相位变化
    unwrapped_phase : 原始解缠绕相位时间序列
    smooth_phase : Savitzky-Golay滤波后的平滑相位
    micro_doppler_phase : 阈值处理前的微多普勒相位变化
    """
    # 在距离维度上找到能量最大的单元
    range_bin_data = find_max_energy_range_bin(data_range_fft)
    
    # 提取并解缠绕相位
    unwrapped_phase, magnitude = extract_and_unwrap_phase(range_bin_data)
    
    # 使用Savitzky-Golay滤波器
    processed_phase, smooth_phase, micro_doppler_phase = process_micro_phase(unwrapped_phase)
    
    return processed_phase, unwrapped_phase, smooth_phase, micro_doppler_phase

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
   maxloc = []
   phase_range = []
   
   # 处理每个时间采样点
   for tn in range(0, data_range_fft.shape[2], time_increment):
       # 在最大能量距离单元附近搜索
       range_slice = np.abs(data_range_fft[channel_num, 
                                         max_range_idx-range_search:max_range_idx+range_search+1, 
                                         tn])
       maxV = np.max(range_slice)
       maxloc_tn = np.argmax(range_slice)
       
       # 处理零值情况，使用前一个位置
       if maxV == 0 and tn > 0:
           maxloc_tn = maxloc[-1]
           
       maxloc.append(maxloc_tn)
       
       # 计算相位索引并提取相位信息
       phase_idx = maxloc_tn + max_range_idx - range_search
       phase_range.append(data_range_fft[channel_num, phase_idx, tn])
   
   # 转换为numpy数组
   phase_range = np.array(phase_range)
   
   return phase_range

