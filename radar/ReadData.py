import numpy as np
import os

def read_dca1000(file_name, num_adc_bits  = 16, num_lanes = 4,is_real = False):
    """
    读取DCA1000雷达数据采集器的数据文件
    
    参数:
    file_name (str): 文件路径
    is_real: 设置为1表示仅实部数据，0表示复数数据
    num_adc_bits: 每个样本的ADC位数
    num_lanes: 通道数量，始终为4，即使只使用1个通道，未使用的通道填充0
    
    返回:
    numpy.ndarray: 处理后的ADC数据
    """
    # 读取.bin文件
    with open(file_name, 'rb') as fid:
        # DCA1000应该以二进制补码形式读取数据
        adc_data = np.fromfile(fid, dtype=np.int16)
    
    # 如果文件名包含'raw'，进行数据包处理
    if 'raw' in file_name:
        packet_len = 1470 // 2  # UDP协议
        full_packets = len(adc_data) // packet_len
        
        # 分离尾部数据
        adc_data_tail = adc_data[packet_len * full_packets:]
        
        # 重塑主要数据部分
        adc_data_main = adc_data[:packet_len * full_packets].reshape(full_packets, packet_len).T
        
        # 移除前7行
        adc_data_main = adc_data_main[7:, :]
        
        # 重新展平
        adc_data_main = adc_data_main.flatten()
        
        # 处理尾部数据并合并
        if len(adc_data_tail) > 7:
            adc_data = np.concatenate([adc_data_main, adc_data_tail[7:]])
        else:
            adc_data = adc_data_main
    
    # 如果ADC位数不是16位，则补偿符号扩展
    if num_adc_bits != 16:
        l_max = 2**(num_adc_bits - 1) - 1
        adc_data[adc_data > l_max] = adc_data[adc_data > l_max] - 2**num_adc_bits
    
    if is_real:
        # 根据每个LVDS通道一个样本重塑数据
        adc_data = adc_data.reshape(num_lanes, -1)
    else:
        # 对于复数据，重塑并组合复数的实部和虚部
        adc_data = adc_data.reshape(num_lanes * 2, -1, order='F')
        real_part = adc_data[0:4, :]
        imag_part = adc_data[4:8, :]
        adc_data = real_part + 1j * imag_part
    
    return adc_data

def radar_params_extract(file_path):
    """
    Extract radar parameters from log file
    
    Parameters:
    -----------
    addr : str
        Directory path containing the radar data files
    raw_file_name : str
        Name of the raw data file
        
    Returns:
    --------
    tuple
        Tuple containing all extracted radar parameters:
        (ADCSample, ChirpPeriod, ADCFs, nchirp_loops, FramPeriod, FramNum, 
         slope, BandWidth, R_Maximum, R_resulo, V_Maximum, V_resulo)
    """
    
    if 'Raw' in file_path:
        para_file_name = f"{file_path[:-10]}_LogFile.txt"
    else:
        para_file_name = f"{file_path[:-4]}_LogFile.txt"
        
    # Read all lines from the parameter file
    with open(para_file_name, 'r') as file:
        para = file.readlines()
    
    # Find lines containing specific configuration strings
    idx1 = []
    idx2 = []
    for i, line in enumerate(para):
        if 'API:ProfileConfig' in line:
            idx1.append(i)
        if 'API:FrameConfig' in line:
            idx2.append(i)
    
    # Split the configuration lines by comma
    pc = para[idx1[-1]].split(',')
    pf = para[idx2[-1]].split(',')
    
    # Extract and calculate parameters
    ADCSample = float(pc[10])  # number
    ChirpPeriod = (float(pc[3]) + float(pc[5])) / 100  # us
    ADCFs = float(pc[11]) * 1000  # sps
    nchirp_loops = float(pf[4])
    FramPeriod = float(pf[5]) / 1e2 / 2  # us
    FramNum = float(pf[3])
    slope = (float(pc[8]) + 2) * 0.048 * 1e12  # Hz/s
    
    fs = 1 / ChirpPeriod * 1e6
    BandWidth = ADCSample / ADCFs * slope
    lamda = 3e8 / 77e9
    R_resulo = 3e8 / 2 / BandWidth  # range resolution
    V_resulo = lamda / 2 / ChirpPeriod * 1e6 / nchirp_loops  # velocity resolution
    
    # Calculate maximum range and velocity
    R_Maximum = ADCFs / slope * 3e8 / 2
    V_Maximum = lamda / 4 * fs
    
    # Return all parameters
    return (ADCSample, ChirpPeriod, ADCFs, nchirp_loops, FramPeriod, FramNum, 
            slope, BandWidth, R_Maximum, R_resulo, V_Maximum, V_resulo)
