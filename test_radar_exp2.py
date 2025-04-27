import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.fft import fft
import sys
import radar
import scipy.signal as signal
from scipy.signal import savgol_filter
import demodulation
import simulation
from math import ceil
from scipy import signal as sp_signal
import modulation

# Parameter setting
fc = 200  # str2double(fileName(6:8))
FrameNum = list(range(1, 257))  # 1:256 in MATLAB
lamda = 3e8 / 77e9
ChannlNum = 0  # Python is 0-indexed, MATLAB is 1-indexed
Rb = 100
fc = 200  # Rb: 码元速率
modulationIndex = fc / Rb  # Modulate at one bit per two cycles

# Load rawData3D
folder = 'data/exp'  # 指定包含.bin文件的文件夹路径
file_pattern = os.path.join(folder, '*BP*SK*.bin')
files = glob.glob(file_pattern)  # 获取文件夹中所有匹配的.bin文件的列表

for file_path in files:
    file_name = os.path.basename(file_path)
    rawData = radar.read_dca1000(file_path)
    
    params = radar.radar_params_extract(file_path)
    ADCSample, ChirpPeriod, ADCFs, ChirpNum, FramPeriod, FramNum, slope, BandWidth, R_Maximum, R_resulo, V_Maximum, V_resulo = params
    fs = 1e6 / ChirpPeriod

    Len = rawData.shape[1]
    fullChirp = FramPeriod / ChirpPeriod
    
    times, times_compen = radar.create_time_arrays(ChirpPeriod, FrameNum, fullChirp)
    
    ChannlNum = 0  
    
    frames_dimension = int(round(Len/(ADCSample*ChirpNum)))
    Data_all = np.reshape(rawData, (4, int(ADCSample), int(ChirpNum), frames_dimension), order='F')
    proData = np.reshape(Data_all[:, :, :, np.array(FrameNum)-1], (4, int(ADCSample), -1), order='F')
    
    DataRangeFft, _ = radar.range_fft(proData, int(ADCSample), BandWidth, apply_window=False)
    _, maxlocAll = radar.find_max_energy_range_bin(DataRangeFft[ChannlNum, :, :])
    phase_range = radar.extract_phase_from_max_range_bin(DataRangeFft, maxlocAll, range_search=3, channel_num=ChannlNum, time_increment=1)
    
    # Process max power range bin 
    unwrapped_phase, _ = radar.extract_and_unwrap_phase(phase_range)
    processed_phase, _, _ = radar.process_micro_phase(unwrapped_phase, times, times_compen, window_size=57, poly_order=3, threshold=0.02)
    
    # BPSK demodulation using two methods
    _, rxData, _, _, raloc = demodulation.bpsk_demodulator_with_symbol_sync(fs, fc, modulationIndex, processed_phase)
    
    # Calculate error rates
    shifted_pattern, error = demodulation.Error110Func(rxData)
    print(f"File: {file_name}, Error Rate: {error}")
    
    num_repetitions = ceil(len(rxData) / 3)
    label, bits = simulation.generate_bpsk_signal(fs, fc, modulationIndex, bit_pattern=shifted_pattern)
    
    input = processed_phase[raloc:raloc+len(label)]
    
    
    
