import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .DataGenerator import generate_random_binary
from .ChannelSimulation import alpha_dist_noise, gaussian_noise
from .OceanWaveSimulation import PM, generate_time_series

import modulation
import numpy as np
import utils




def micro_wave(num_bits, poly, m_psk, n_fft, n_cp, sample_rate, snr_db, wind_speed):
    # 生成随机二进制数据（label1-bits）
    bits = generate_random_binary(num_bits)

    # 编码后的二进制数据（label2-crc_bits）
    crc_bits = modulation.add_crc(bits, poly)

    # mpsk后的无损编码频域谱（label3-mpsk）
    complex_symbols = modulation.mpsk_modulation(crc_bits, m_psk)

    # ofdm编码
    ofdm_signal, frame_structure = modulation.ofdm_modulation(complex_symbols, n_fft, n_cp, pilot_pattern='comb', comb_num=8)
    real_ofdm_signal = np.real(ofdm_signal)
    time_indices = modulation.generate_time_indices(real_ofdm_signal, sample_rate, 0)
    
    # 信道模拟
    alpha_noise = alpha_dist_noise(real_ofdm_signal, 1.5, 0, 1, 0, snr_db)
    gaussain_noise = gaussian_noise(real_ofdm_signal, snr_db)
    noised_signal = alpha_noise + gaussain_noise + real_ofdm_signal
    
    # 微幅波传导后时域谱（input1-normalized_signal）
    normalised_signal = utils.signal_normalize(noised_signal)
    
    # 添加有限振幅波
    duration = np.max(time_indices)
    dt = time_indices[1] - time_indices[0]
    w, S = PM(wind_speed)
    t, eta = generate_time_series(w, S, duration, dt)
    
    # 确保eta和normalized_signal长度匹配
    min_length = min(len(eta), len(normalised_signal))
    eta = eta[:min_length]
    normalised_signal = normalised_signal[:min_length]
    
    # 增加有限振幅波后时域谱（input2-recieved_signal）    
    eta_signal = normalised_signal + eta
    
    return bits, crc_bits, noised_signal, eta_signal, frame_structure




# 水下衰减









