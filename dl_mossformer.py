import deeplearning as dl
import radar
import demodulation
import utils
from deeplearning.model_configs import mossformer2_librimix_2spk, sigsep_microwave
import pandas as pd
import numpy as np

config = 'train' # 'train' or 'predict'
method = 'sigsep' # 'mossformer2' or 'sigsep'

save_dir = "model/mossformer2"
channel_est_pt = 'model/mossformer2/best_model.pt'
structure_dir = 'data/train_frame_structure.json'


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    if config == "train":
        
        if method == 'mossformer2':

            dl.train_mossformer2(config=mossformer2_librimix_2spk, num_epochs=120, save_dir=save_dir)
            
        if method == 'sigsep':
            
            dl.train_sigsep(config=sigsep_microwave , num_epochs=120, save_dir=save_dir)
 

        
    elif config =="predict":
        
        frame_structure = pd.read_json(structure_dir).to_dict()
        for key in ['n_fft', 'n_cp', 'num_ofdm_symbols']:
            if key in frame_structure and isinstance(frame_structure[key], dict):
                frame_structure[key] = list(frame_structure[key].values())[0]

        if method == 'mossformer2':
            # 加载测试数据（含噪声信号）
            test_data = utils.load_data_from_csv('data/Val_recieved_signal/0000.csv')
            
            # 加载原始干净信号以获取原始比特流
            original_signal_data = utils.load_data_from_csv('data/Val_ofdm_signal&ocean_wave/0000.csv', columns='ofdm_signal')
            original_signal_data = original_signal_data.reshape(-1)
            
            # 使用Mossformer2模型进行预测
            est_result = dl.predict_mossformer2(channel_est_pt, test_data)
            est_result = est_result.squeeze(0).cpu().numpy()
            
            # 获取需要的信号
            noised_signal = est_result[:,0]  # 测试数据（含噪声）
            
            # OFDM解调原始干净信号以获取原始比特
            orig_freq_symbols, orig_symbol_mapping = demodulation.ofdm_preprocessing(original_signal_data, frame_structure)
            
            # 确保n_fft是整数而非字典
            n_fft = frame_structure['n_fft']
            if isinstance(n_fft, dict):
                n_fft = list(n_fft.values())[0]
            
            # 信道估计和均衡（对于原始干净信号，可能不需要复杂的信道估计）
            # 直接使用理想信道估计（全1）
            ideal_channel = np.ones_like(orig_freq_symbols)
            orig_demodulated_symbols = demodulation.apply_equalization(orig_freq_symbols, ideal_channel, orig_symbol_mapping, 'zf')
            
            # 确保num_bits已定义或从frame_structure获取
            if 'num_bits' in frame_structure:
                num_bits = frame_structure['num_bits']
                if isinstance(num_bits, dict):
                    num_bits = list(num_bits.values())[0]
            else:
                # 如果无法确定num_bits，可以尝试从demodulated_symbols推断
                num_bits = len(orig_demodulated_symbols) * 2  # 假设QPSK调制(每个符号2比特)
            
            # QPSK解调得到原始比特
            original_bits, orig_decoded_symbols = demodulation.mpsk_demodulation(orig_demodulated_symbols[:num_bits//2], 4)
            
            print(f"成功从原始信号解调出 {len(original_bits)} 比特")
            
            # OFDM解调含噪声信号
            ofdm_freq_symbols, symbol_mapping = demodulation.ofdm_preprocessing(noised_signal, frame_structure)

            # 信道估计和均衡
            channel_estimates = demodulation.estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)
            demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, channel_estimates, symbol_mapping, 'zf')
            
            # QPSK解调
            demodulated_bits, decoded_symbols = demodulation.mpsk_demodulation(demodulated_symbols[:num_bits//2], 4)
            
            # 现在可以使用demodulated_bits进行后续处理...
            print(f"成功从含噪声信号解调出 {len(demodulated_bits)} 比特")
            
            # 计算比特错误率（如果有原始比特序列）
            if original_bits is not None:
                # 确保比较相同长度的比特序列
                min_len = min(len(demodulated_bits), len(original_bits))
                bit_errors = np.sum(demodulated_bits[:min_len] != original_bits[:min_len])
                ber = bit_errors / min_len
                print(f"比特错误率: {ber:.6f} ({bit_errors}/{min_len})")
            
        


