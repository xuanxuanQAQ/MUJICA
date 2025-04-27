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

num_bits = 2048
poly = 'CRC-16'
m_psk = 4
n_fft = 64
comb_num = 8
n_cp = 16
sample_rate = 150
snr_db = 20
wind_speed = 1.5

save_dir = "model/channel_estimation"
channel_est_pt = 'model/channel_estimation/best_model.pth'

if config == "train":
    # Load the data
    data = dl.load_channel_estimation_data(
            mpsk_label_dir='data/Train_mpsk_signal_B01', 
            recieved_signal_dir='data/Train_ofdm_freq_symbols_B01',
            structure_dir='data/train_frame_structure.json',
            batch_size=4
        )

    dl.train_channel_estimation(model, data, num_epochs=30, learning_rate=0.001, save_dir=save_dir)
    
elif config =="predict":
    
    bits, crc_bits, noised_signal, eta_signal, frame_structure = simulation.micro_wave(num_bits, poly, m_psk, n_fft, n_cp, sample_rate, snr_db, wind_speed)

    # 方法一：深度学习
    ofdm_freq_symbols, symbol_mapping = demodulation.ofdm_preprocessing(noised_signal, frame_structure)

    dl_channel_estimates = dl.predict_channel_estimation(model, ofdm_freq_symbols, symbol_mapping, 
                                    frame_structure['num_ofdm_symbols'] * frame_structure['n_fft'], channel_est_pt)
    dl_channel_estimates = np.squeeze(dl_channel_estimates, axis=0)
    dl_channel_estimates = dl_channel_estimates[:, 0] + 1j * dl_channel_estimates[:, 1]

    dl_demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, dl_channel_estimates, symbol_mapping, 'zf')
    dl_demodulated_bits, dl_decoded_symbols = demodulation.mpsk_demodulation(dl_demodulated_symbols[:num_bits//2], 4)

    # 计算深度学习方法的误比特率
    dl_bit_errors = np.sum(bits[:len(dl_demodulated_bits)] != dl_demodulated_bits)
    dl_ber = dl_bit_errors / len(dl_demodulated_bits)

    # 方法二：插值估计
    interp_channel_estimates = demodulation.estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)

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
    
