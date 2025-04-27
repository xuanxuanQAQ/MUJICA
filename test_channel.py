import simulation
import modulation
import plot
import demodulation
import utils
import numpy as np

# 参数设置
# 数据相关参数
num_bits = 2048             # 数据比特数
n_fft = 64                  # FFT大小
n_cp = 16                   # 循环前缀长度
num_data_carriers = 52      # 数据载波数量
poly = 'CRC-16'             # CRC编码方式
m_psk = 4                   # M-PSK的M

# 模拟相关参数
sample_rate = 100           # 采样率
snr_db = 50                 # 噪声信噪比(dB)
wind_speed = 1              # 风速(m/s)
fetch_length = 1e6          # 风区长度(m)

# 雷达设置相关参数
ADCSample = 256             # ADC采样点数，单个chirp的采样数量
ka = 60.012e12              # 调频斜率(Hz/s)，表示频率随时间变化的速率
Fadc = 5e6                  # ADC采样频率(Hz)
ChirpPeriod = 100e-6        # 单个chirp的周期(s)
rampTime = 66e-6            # 调频时间(s)，即频率从最低上升到最高所需时间
ChirpNum = 255              # 每帧chirp数量
FramPeriod = 25.6e-3        # 帧周期(s)，一帧完成所需时间
FramNum = 1024              # 总帧数
BandWidth = ADCSample/Fadc*ka    # 信号带宽(Hz)，决定距离分辨率
lamda = 3e8/77e9            # 雷达波长(m)，基于77GHz毫米波雷达
R_resulo = 3e8/2/BandWidth  # 距离分辨率(m)，雷达能够分辨的最小距离差
fs = 1/ChirpPeriod          # 帧采样率(Hz)
f_resulo = ka*2*R_resulo/3e8 # 频率分辨率(Hz)
f_deta = Fadc/256           # 频率间隔(Hz)
nfft_d = (ADCSample)        # 距离维FFT点数
nfft_v = (ChirpNum)         # 速度维FFT点数
nfft_f = (FramNum)          # 帧维FFT点数



# 生成随机数字信号
bits = simulation.generate_random_binary(num_bits)

# CRC
crc_bits = modulation.add_crc(bits, poly)

# QPSK
complex_symbols = modulation.mpsk_modulation(crc_bits, m_psk)

# 绘制星座图
plot.plot_constellation(complex_symbols, 'QPSK 星座图')

# OFDM
ofdm_signal, frame_structure = modulation.ofdm_modulation(complex_symbols, n_fft, n_cp, pilot_pattern='comb', comb_num=8)
real_ofdm_signal = np.real(ofdm_signal)

# 绘制OFDM信号时域和频域波形
plot.plot_ofdm_signal_analysis(real_ofdm_signal, n_fft, sample_rate)
time_indices = modulation.generate_time_indices(real_ofdm_signal, sample_rate, 0)
 
# 添加高斯白噪声，模拟信道影响
alpha_dist_noise = simulation.alpha_dist_noise(real_ofdm_signal, 1.5,  0, 1, 0, snr_db)
gaussain_noise = simulation.gaussian_noise(real_ofdm_signal, snr_db)
noised_signal = alpha_dist_noise + gaussain_noise + real_ofdm_signal

normalised_signal = utils.signal_normalize(noised_signal)

# 添加有限振幅波
duration = np.max(time_indices)
dt = time_indices[1] - time_indices[0]
w, S = simulation.PM(wind_speed)
t, eta = simulation.generate_time_series(w, S, duration, dt)
plot.plot_spectrum_and_time_series(w, S, t, eta, wind_speed)

recieved_signal = normalised_signal + eta
plot.plot_microamplitude_wave(t, recieved_signal, wind_speed)

# OFDM解调
demodulated_symbols, channel_estimates = demodulation.ofdm_demodulation(normalised_signal, frame_structure)

# 绘制解调后的星座图
plot.plot_constellation(demodulated_symbols, f'解调后的星座图 (SNR={snr_db}dB)')

# QPSK解调
demodulated_bits, decoded_symbols = demodulation.mpsk_demodulation(demodulated_symbols[:num_bits//2], 4)

# 计算误比特率
bit_errors = np.sum(bits[:len(demodulated_bits)] != demodulated_bits)
ber = bit_errors / len(demodulated_bits)
print(f"比特错误数: {bit_errors}")
print(f"误比特率 (BER): {ber:.6f}")

# CRC验证
is_valid = demodulation.crc_check(demodulated_bits, poly, crc_bits[len(bits):])
print(f"CRC校验结果: {'通过' if is_valid else '失败'}")

input()
