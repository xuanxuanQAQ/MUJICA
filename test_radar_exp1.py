import radar
import numpy as np
from scipy import signal
import os
import plot
import radar.RangeFFT


ADCSample = 256             # ADC采样点数，单个chirp的采样数量
ka = 60.012e12              # 调频斜率(Hz/s)，表示频率随时间变化的速率
Fadc = 5e6                  # ADC采样频率(Hz)
ChirpPeriod = 100e-6        # 单个chirp的周期(s)
rampTime = 66e-6            # 调频时间(s)，即频率从最低上升到最高所需时间
ChirpNum = 255              # 每帧chirp数量
FramPeriod = 25.6e-3        # 帧周期(s)，一帧完成所需时间
FramNum = 1024              # 总帧数
BandWidth = ADCSample/Fadc*ka    # 信号带宽(Hz)，决定距离分辨率
fc = 77e9                   # 载波频率
lamda = 3e8/fc              # 雷达波长(m)，基于77GHz毫米波雷达
R_resulo = 3e8/2/BandWidth  # 距离分辨率(m)，雷达能够分辨的最小距离差
fs = 1/ChirpPeriod          # 帧采样率(Hz)
f_resulo = ka*2*R_resulo/3e8 # 频率分辨率(Hz)
f_deta = Fadc/256           # 频率间隔(Hz)
nfft_d = (ADCSample)        # 距离维FFT点数
nfft_v = (ChirpNum)         # 速度维FFT点数
nfft_f = (FramNum)          # 帧维FFT点数
num_adc_bits  = 16          # 每个样本的ADC位数
num_lanes = 4               # 通道数量


addr = 'data/exp'
raw_file_name = 'BFPSKRb100A110Fc200P20F2_S1_Raw_0.bin'
file_path = os.path.join(addr, raw_file_name)

adc_data = radar.read_dca1000(file_path, num_adc_bits, num_lanes)


for ChannlNum in [1]:   # 接收通道选择,1-4
    FrameNum = np.arange(1, 257)
    
    Data_all = np.reshape(adc_data[ChannlNum-1, :], (ADCSample, ChirpNum, -1), order='F')
    
    # range FFT
    proData = np.reshape(Data_all[:, :, FrameNum-1], (ADCSample, ChirpNum*len(FrameNum)), order='F')
    DataRangeFft, range_axis = radar.range_fft(proData, ADCSample, BandWidth)
    plot.plot_range_fft(DataRangeFft, range_axis, os.path.splitext(file_path)[0])
    plot.plot_range_profile(DataRangeFft, range_axis)

    # 精细相位分析，有BUG
    processed_phase, unwrapped_phase, smooth_phase, sub_phase = radar.process_micromotion_phase(DataRangeFft)
    plot.plot_micromotion_phase(unwrapped_phase, smooth_phase, sub_phase, processed_phase, ChirpPeriod, FrameNum)
    
    # STFT分析 ，有BUG
    # f, t, Zxx = radar.phase_stft(processed_phase, fs)
    # plot.plot_stft(f, t, Zxx)

    
input()




