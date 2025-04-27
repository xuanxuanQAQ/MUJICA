import deeplearning as dl
import simulation
import modulation
import utils
import demodulation
import numpy as np
import os
import torch

config = 'train'  # 'train' or 'predict'

num_bits = 2048
poly = 'CRC-16'
m_psk = 4
n_fft = 64
comb_num = 8
n_cp = 16
sample_rate = 150
snr_db = 20
wind_speed = 1.5

epochs = 250
learning_rate=0.001

save_dir = 'model/channel_equalizer'
structure_dir = 'data/train_frame_structure.json'
channel_est_pt = 'model/channel_estimation/best_model.pth'
channel_eql_pt = 'model/channel_equalizer/best_model.pth'

model = dl.ChannelEqualizerNet()

if config == "train":
    train_loader = dl.load_channel_equalizer_data(recieved_signal_dir='data/Train_ofdm_freq_symbols_B01',
                                channel_est_dir='data/Train_channel_estimates_B01',
                                mpsk_label_dir='data/Train_mpsk_signal_B01', 
                                structure_dir=structure_dir,
                                shuffle=True,
                                num_workers=0)

    model = dl.train_channel_equalizer(model, train_loader, epochs, learning_rate, save_dir=save_dir, structure_dir=structure_dir)
    
elif config =="predict":
    bits, crc_bits, noised_signal, eta_signal, frame_structure = simulation.micro_wave(num_bits, poly, m_psk, n_fft, n_cp, sample_rate, snr_db, wind_speed)
    ofdm_freq_symbols, symbol_mapping = demodulation.ofdm_preprocessing(noised_signal, frame_structure)
    
    est_model = dl.Transformer(
        input_dim=2,  # 复数信号 (实部+虚部)
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    # 方法一：深度学习信道估计 + 深度学习信道均衡
    print("\n===== 方法一：深度学习信道估计 + 深度学习信道均衡 =====")
    
    # 进行深度学习信道估计
    dl_channel_estimates = dl.predict_channel_estimation(est_model, ofdm_freq_symbols, symbol_mapping, frame_structure['num_ofdm_symbols'] * frame_structure['n_fft'], channel_est_pt)
    
    # 将通道估计结果从实部和虚部转换为复数形式
    dl_channel_estimates = np.squeeze(dl_channel_estimates, axis=0)
    dl_complex_channel_estimates = dl_channel_estimates[:, 0] + 1j * dl_channel_estimates[:, 1]
    
    # 准备输入数据格式
    # 将复数转换为实部和虚部的格式 [H, W, 2]
    ofdm_real_imag = np.zeros((ofdm_freq_symbols.shape[0], ofdm_freq_symbols.shape[1], 2), dtype=np.float32)
    ofdm_real_imag[:, :, 0] = np.real(ofdm_freq_symbols)
    ofdm_real_imag[:, :, 1] = np.imag(ofdm_freq_symbols)
    
    channel_real_imag = np.zeros((ofdm_freq_symbols.shape[0], ofdm_freq_symbols.shape[1], 2), dtype=np.float32)
    dl_channel_estimates = dl_channel_estimates.reshape(ofdm_freq_symbols.shape[0], ofdm_freq_symbols.shape[1], 2)
    
    channel_real_imag[:, :, 0] = dl_channel_estimates[:, :, 0].cpu().numpy().astype(np.float32)
    channel_real_imag[:, :, 1] = dl_channel_estimates[:, :, 1].cpu().numpy().astype(np.float32)
    
    # 添加batch维度
    channel_real_imag = np.expand_dims(channel_real_imag, axis=0)  # [1, H, W, 2]
    ofdm_real_imag = np.expand_dims(ofdm_real_imag, axis=0)  # [1, H, W, 2]
    
    # 使用深度学习信道均衡进行预测
    dl_demodulated_symbols = dl.predict_channel_equalizer(model, channel_real_imag, ofdm_real_imag, structure_dir, channel_eql_pt)
    dl_demodulated_symbols= dl_demodulated_symbols.squeeze()
    # 将均衡结果从实部和虚部转换为复数形式
    dl_complex_demodulated_symbols = dl_demodulated_symbols[:, 0] + 1j * dl_demodulated_symbols[:, 1]
    
    # 解调信号
    dl_demodulated_bits, dl_decoded_symbols = demodulation.mpsk_demodulation(
        dl_complex_demodulated_symbols[:num_bits//2], m_psk
    )
    
    # 方法二：传统插值信道估计 + 深度学习信道均衡
    print("\n===== 方法二：传统插值信道估计 + 深度学习信道均衡 =====")
    
    # 使用传统方法进行信道估计
    interp_channel_estimates = demodulation.estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)
    
    # 准备传统插值信道估计的输入数据格式
    # 将复数转换为实部和虚部的格式 [H, W, 2]
    interp_channel_real_imag = np.zeros((ofdm_freq_symbols.shape[0], ofdm_freq_symbols.shape[1], 2), dtype=np.float32)
    interp_channel_real_imag[:, :, 0] = np.real(interp_channel_estimates).reshape(ofdm_freq_symbols.shape[0], ofdm_freq_symbols.shape[1])
    interp_channel_real_imag[:, :, 1] = np.imag(interp_channel_estimates).reshape(ofdm_freq_symbols.shape[0], ofdm_freq_symbols.shape[1])
    
    # 添加batch维度
    interp_channel_real_imag = np.expand_dims(interp_channel_real_imag, axis=0)  # [1, H, W, 2]
    
    # 使用深度学习信道均衡进行预测
    dl_with_interp_est_demodulated_symbols = dl.predict_channel_equalizer(
        model, interp_channel_real_imag, ofdm_real_imag, structure_dir, channel_eql_pt
    )
    dl_with_interp_est_demodulated_symbols = dl_with_interp_est_demodulated_symbols.squeeze()
    
    # 将均衡结果从实部和虚部转换为复数形式
    dl_with_interp_est_complex_demodulated_symbols = dl_with_interp_est_demodulated_symbols[:, 0] + 1j * dl_with_interp_est_demodulated_symbols[:, 1]
    
    # 解调信号
    dl_with_interp_est_demodulated_bits, dl_with_interp_est_decoded_symbols = demodulation.mpsk_demodulation(
        dl_with_interp_est_complex_demodulated_symbols[:num_bits//2], m_psk
    )
    
    # 方法三：传统插值信道估计 + 传统ZF均衡
    print("\n===== 方法三：传统插值信道估计 + 传统ZF均衡 =====")
    
    # 使用传统ZF均衡器（复用方法二的传统信道估计结果）
    interp_demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, interp_channel_estimates, symbol_mapping, 'zf')
    interp_demodulated_bits, interp_decoded_symbols = demodulation.mpsk_demodulation(interp_demodulated_symbols[:num_bits//2], m_psk)
    
    # 计算各种方法的误比特率
    dl_bit_errors = np.sum(bits[:len(dl_demodulated_bits)] != dl_demodulated_bits)
    dl_ber = dl_bit_errors / len(dl_demodulated_bits)
    
    dl_with_interp_est_bit_errors = np.sum(bits[:len(dl_with_interp_est_demodulated_bits)] != dl_with_interp_est_demodulated_bits)
    dl_with_interp_est_ber = dl_with_interp_est_bit_errors / len(dl_with_interp_est_demodulated_bits)
    
    interp_bit_errors = np.sum(bits[:len(interp_demodulated_bits)] != interp_demodulated_bits)
    interp_ber = interp_bit_errors / len(interp_demodulated_bits)
    
    # 打印结果对比
    print("\n===== 结果对比 =====")
    print(f"方法一 (DL估计+DL均衡) - 比特错误数: {dl_bit_errors}, 误比特率 (BER): {dl_ber:.6f}")
    print(f"方法二 (传统估计+DL均衡) - 比特错误数: {dl_with_interp_est_bit_errors}, 误比特率 (BER): {dl_with_interp_est_ber:.6f}")
    print(f"方法三 (传统估计+ZF均衡) - 比特错误数: {interp_bit_errors}, 误比特率 (BER): {interp_ber:.6f}")
    
    # 计算性能提升
    if interp_ber > 0:
        dl_improvement = ((interp_ber - dl_ber) / interp_ber) * 100
        dl_with_interp_est_improvement = ((interp_ber - dl_with_interp_est_ber) / interp_ber) * 100
        print(f"方法一相对于方法三的性能提升: {dl_improvement:.2f}%")
        print(f"方法二相对于方法三的性能提升: {dl_with_interp_est_improvement:.2f}%")
    
    # CRC验证结果对比
    dl_is_valid = demodulation.crc_check(dl_demodulated_bits, poly, crc_bits[len(bits):])
    dl_with_interp_est_is_valid = demodulation.crc_check(dl_with_interp_est_demodulated_bits, poly, crc_bits[len(bits):])
    interp_is_valid = demodulation.crc_check(interp_demodulated_bits, poly, crc_bits[len(bits):])
    
    print(f"方法一 CRC校验结果: {'通过' if dl_is_valid else '失败'}")
    print(f"方法二 CRC校验结果: {'通过' if dl_with_interp_est_is_valid else '失败'}")
    print(f"方法三 CRC校验结果: {'通过' if interp_is_valid else '失败'}")
