import deeplearning as dl
import simulation
import modulation
import utils
import demodulation
import numpy as np

# import multiprocessing
# multiprocessing.freeze_support()

model = dl.Transformer(
        input_dim=2,  # 复数信号 (实部+虚部)
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3
    )

config = 'predict'

num_bits=2048
poly='CRC-16'
m_psk=4
n_fft=64
comb_num=8
n_cp=16
sample_rate=150
snr_db=20
wind_speed=1.5

if config == "train":
    # Load the data
    data = dl.load_channel_data(
            mpsk_label_dir='data/Train_mpsk_signal_B01', 
            recieved_signal_dir='data/Train_ofdm_freq_symbols_B01',
            structure_dir='data/train_frame_structure.json',
            batch_size=4
        )

    dl.train_model(model, data, num_epochs=30, learning_rate=0.001, save_dir="model")
    
elif config =="predict":
    
    # 生成随机二进制数据（label1-bits）
    bits = simulation.generate_random_binary(num_bits)

    # 编码后的二进制数据（label2-crc_bits）
    crc_bits = modulation.add_crc(bits, poly)

    # mpsk后的无损编码频域谱（label3-mpsk）
    complex_symbols = modulation.mpsk_modulation(crc_bits, m_psk)

    # ofdm编码
    ofdm_signal, frame_structure = modulation.ofdm_modulation(complex_symbols, n_fft, n_cp, pilot_pattern='comb', comb_num=8)
    real_ofdm_signal = np.real(ofdm_signal)
    time_indices = modulation.generate_time_indices(real_ofdm_signal, sample_rate, 0)
    
    # 信道模拟
    alpha_dist_noise = simulation.alpha_dist_noise(real_ofdm_signal, 1.5, 0, 1, 0, snr_db)
    gaussain_noise = simulation.gaussian_noise(real_ofdm_signal, snr_db)
    noised_signal = alpha_dist_noise + gaussain_noise + real_ofdm_signal
    
    # 微幅波传导后时域谱（input1-normalized_signal）
    normalised_signal = utils.signal_normalize(noised_signal)
    
    # 添加有限振幅波
    duration = np.max(time_indices)
    dt = time_indices[1] - time_indices[0]
    w, S = simulation.PM(wind_speed)
    t, eta = simulation.generate_time_series(w, S, duration, dt)
    
    # 确保eta和normalized_signal长度匹配
    min_length = min(len(eta), len(normalised_signal))
    eta = eta[:min_length]
    normalised_signal = normalised_signal[:min_length]
    
    # 增加有限振幅波后时域谱（input2-recieved_signal）    
    recieved_signal = normalised_signal + eta

    # 方法一：深度学习
    ofdm_freq_symbols, symbol_mapping = demodulation.ofdm_preprocessing(noised_signal, frame_structure)

    dl_channel_estimates = dl.predict(model, ofdm_freq_symbols, symbol_mapping, 
                                    frame_structure['num_ofdm_symbols'] * frame_structure['n_fft'], 
                                    'model/best_model.pth')
    dl_channel_estimates = np.squeeze(dl_channel_estimates, axis=0)
    dl_channel_estimates = dl_channel_estimates[:, 0] + 1j * dl_channel_estimates[:, 1]

    # 使用深度学习估计的信道进行解调
    dl_demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, dl_channel_estimates, symbol_mapping, 'zf')
    dl_demodulated_bits, dl_decoded_symbols = demodulation.mpsk_demodulation(dl_demodulated_symbols[:num_bits//2], 4)

    # 计算深度学习方法的误比特率
    dl_bit_errors = np.sum(bits[:len(dl_demodulated_bits)] != dl_demodulated_bits)
    dl_ber = dl_bit_errors / len(dl_demodulated_bits)

    # 方法二：插值估计
    interp_channel_estimates = demodulation.estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)

    # 使用插值估计的信道进行解调
    interp_demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, interp_channel_estimates, symbol_mapping, 'zf')
    interp_demodulated_bits, interp_decoded_symbols = demodulation.mpsk_demodulation(interp_demodulated_symbols[:num_bits//2], 4)

    # 计算插值方法的误比特率
    interp_bit_errors = np.sum(bits[:len(interp_demodulated_bits)] != interp_demodulated_bits)
    interp_ber = interp_bit_errors / len(interp_demodulated_bits)

    # 打印结果对比
    print("\n===== 信道估计方法对比 =====")
    print(f"深度学习方法 - 比特错误数: {dl_bit_errors}, 误比特率 (BER): {dl_ber:.6f}")
    print(f"插值估计方法 - 比特错误数: {interp_bit_errors}, 误比特率 (BER): {interp_ber:.6f}")
    print(f"性能提升比例: {((interp_ber - dl_ber) / interp_ber) * 100:.2f}% ")

    # CRC验证结果对比
    dl_is_valid = demodulation.crc_check(dl_demodulated_bits, poly, crc_bits[len(bits):])
    interp_is_valid = demodulation.crc_check(interp_demodulated_bits, poly, crc_bits[len(bits):])
    print(f"深度学习方法 CRC校验结果: {'通过' if dl_is_valid else '失败'}")
    print(f"插值估计方法 CRC校验结果: {'通过' if interp_is_valid else '失败'}")
    
