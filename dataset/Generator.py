import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import simulation
import modulation
import os
import time
from tqdm import tqdm
import h5py
from concurrent.futures import ProcessPoolExecutor
import functools
import utils
import demodulation
import glob
import shutil
import json

def generate_batch_data(num_bits=2048, poly='CRC-16', m_psk=4, n_fft=64, comb_num=8, n_cp=16, 
                        sample_rate=150, snr_db=20, wind_speed=1.5, label=None, input=None):
    """
    生成指定数量的批量数据，将标签和输入数据分开存储
    
    参数:
    batch_size: 批量大小
    num_bits: 比特数
    poly: CRC多项式
    m_psk: MPSK调制阶数
    n_fft: FFT点数
    comb_num: 一个symbol内导频数量
    n_cp: 循环前缀长度
    sample_rate: 采样率
    snr_db: 信噪比(dB)
    wind_speed: 风速(m/s)
    label: 标签选择参数，可以是 'bits', 'crc_bits', 'mpsk_signal' 或列表组合
    input: 输入选择参数，可以是 'normalized_signal', 'recieved_signal', 'pilot' 或列表组合
    
    返回:
    sample_labels: 标签数据列表
    sample_inputs: 输入数据列表
    """
    
    # 临时存储当前样本的标签和输入数据
    sample_labels = {}
    sample_inputs = {}
    
    # 生成随机二进制数据（label1-bits）
    bits = simulation.generate_random_binary(num_bits)
    if label == 'bits' or (isinstance(label, list) and 'bits' in label):
        sample_labels['bits'] = bits
    
    # 编码后的二进制数据（label2-crc_bits）
    crc_bits = modulation.add_crc(bits, poly)
    if label == 'crc_bits' or (isinstance(label, list) and 'crc_bits' in label):
        sample_labels['crc_bits'] = crc_bits
    
    # mpsk后的无损编码频域谱（label3-mpsk）
    complex_symbols = modulation.mpsk_modulation(crc_bits, m_psk)
    if label == 'mpsk_signal' or (isinstance(label, list) and 'mpsk_signal' in label):
        sample_labels['mpsk_signal'] = complex_symbols
    
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
    if input == 'normalized_signal' or (isinstance(input, list) and 'normalized_signal' in input):
        sample_inputs['normalized_signal'] = normalised_signal
    
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
    if input == 'recieved_signal' or (isinstance(input, list) and 'recieved_signal' in input):
        sample_inputs['recieved_signal'] = recieved_signal
    
    # OFDM解调
    ofdm_freq_symbols, symbol_mapping = demodulation.ofdm_preprocessing(noised_signal, frame_structure)
    if input == 'ofdm_freq_symbols' or (isinstance(input, list) and 'ofdm_freq_symbols' in input):
        sample_inputs['ofdm_freq_symbols'] = ofdm_freq_symbols.flatten()
        
    n_fft = frame_structure['n_fft']
    channel_estimates = demodulation.estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)
    demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, channel_estimates, symbol_mapping, 'zf')
    
    # QPSK解调（如果需要可以添加到输出）
    demodulated_bits, decoded_symbols = demodulation.mpsk_demodulation(demodulated_symbols[:num_bits//2], 4)
    
    return sample_labels, sample_inputs, frame_structure
    
def get_dataset_paths(root_dir, dataset_type, num_batch, label, input):
    """
    创建数据集存储路径和目录
    
    参数:
    root_dir: 根目录
    dataset_type: 数据集类型 ('train', 'validation', 'test')
    num_batch: 批次数量
    label: 标签类型 (例如 'bits', 'crc_bits', 'mpsk')
    input: 输入类型 (例如 'normalized_signal', 'recieved_signal')
    
    返回:
    包含路径信息的字典
    """
    # 确定数据集类型前缀和分母值
    if dataset_type == 'train':
        prefix = 'Train'
        denom = 100
    elif dataset_type == 'validation':
        prefix = 'Val'
        denom = 10
    else:  # test
        prefix = 'Test'
        denom = 10
    
    # 构建目录名，加入label和input信息
    suffix = f'B{num_batch//denom:02d}'
    
    # 将label和input添加到目录名中
    feature_dir = os.path.join(root_dir, f'{prefix}_{input}_{suffix}')
    label_dir = os.path.join(root_dir, f'{prefix}_{label}_{suffix}')
    
    # 确保目录存在
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # 生成文件前缀
    feature_prefix = f'{prefix}_{input}_{suffix}'
    label_prefix = f'{prefix}_{label}_{suffix}'
    
    return {
        'feature_dir': feature_dir,
        'label_dir': label_dir,
        'feature_prefix': feature_prefix,
        'label_prefix': label_prefix,
    }
    
def generate_train_data(num_batch=1, root_dir="data/", dataset_type="train", num_bits=2048, poly='CRC-16', m_psk=4, n_fft=64, comb_num=8,
                        n_cp=16, sample_rate=150, snr_db=50, wind_speed=1.5, label="bits", input="normalized_signal", use_hdf5=False, use_multiprocessing=False):
    """
    生成训练、验证或测试数据
    
    参数:
    - num_batch: 批次总数
    - comb_num: 一个symbol内导频间隔
    - root_dir: 存储数据集的根目录
    - dataset_type: 数据集类型（train/validation/test）
    - use_hdf5: 是否使用HDF5格式保存（与CSV相比更高效）
    - use_multiprocessing: 是否使用多进程加速数据生成
    
    返回:
    - 无返回值，数据保存到指定目录
    """
    # 验证数据集类型
    valid_types = ['train', 'validation', 'test']
    if dataset_type not in valid_types:
        raise ValueError(f'dataset_type必须是{valid_types}之一')
    
    # 获取数据集存储路径信息
    paths = get_dataset_paths(root_dir, dataset_type, num_batch, label, input)
    
    start_time = time.time()
    
    if use_multiprocessing and num_batch > 1:
        # 使用多进程并行生成数据
        partial_gen = functools.partial(
            generate_batch_data, 
        )
        
        # 使用ProcessPoolExecutor并行处理
        with ProcessPoolExecutor() as executor:
            with tqdm(total=num_batch) as pbar:
                for batch_idx, (batch_Y, batch_X, frame_structure) in enumerate(
                    executor.map(partial_gen, range(num_batch))
                ):
                    save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5)
                    pbar.update(1)
    else:
        # 顺序生成数据
        for batch_idx in tqdm(range(num_batch)):
            batch_Y, batch_X, frame_structure = generate_batch_data(num_bits, poly, m_psk, n_fft, comb_num, n_cp, sample_rate, snr_db, wind_speed, label, input)
            save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5)
    
    # Save frame_structure as JSON
    if num_batch > 0:  # Only save if we generated at least one batch
        frame_structure_path = os.path.join(root_dir, f'{dataset_type}_frame_structure.json')
        with open(frame_structure_path, 'w') as f:
            json.dump(frame_structure, f, indent=4)
    
    end_time = time.time()
    print(f'生成并保存{num_batch}个批次耗时: {end_time-start_time:.2f}秒')
    print(f'数据已保存到目录: {os.path.abspath(root_dir)}')
    
    
def save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5=False):
    """
    保存单个批次的数据，支持CSV或HDF5格式
    对于复数数据，将实部和虚部分别存储在CSV的相邻两行中
    一行存储实部，下一行存储虚部，每两行表示一个完整的数据
    
    参数:
    batch_idx: 批次索引
    batch_Y: 单个批次的标签数据
    batch_X: 单个批次的特征数据
    paths: 保存路径字典
    use_hdf5: 是否使用HDF5格式保存
    """
    feature_path = os.path.join(paths['feature_dir'], f'{batch_idx:04d}')
    label_path = os.path.join(paths['label_dir'], f'{batch_idx:04d}')
    
    if use_hdf5:
        # 使用HDF5格式保存
        with h5py.File(f'{feature_path}.h5', 'w') as f:
            # 这里直接处理单个批次的特征数据
            key = list(batch_X.keys())[0]
            data = batch_X[key]
            
            # 检查是否包含复数，如果是则分离实部和虚部
            if np.iscomplexobj(data):
                f.create_dataset('real', data=np.real(data))
                f.create_dataset('imag', data=np.imag(data))
            else:
                f.create_dataset('data', data=data)
        
        with h5py.File(f'{label_path}.h5', 'w') as f:
            # 这里直接处理单个批次的标签数据
            key = list(batch_Y.keys())[0]
            data = batch_Y[key]
            
            # 检查是否包含复数，如果是则分离实部和虚部
            if np.iscomplexobj(data):
                f.create_dataset('real', data=np.real(data))
                f.create_dataset('imag', data=np.imag(data))
            else:
                f.create_dataset('data', data=data)
    else:
        # 处理CSV格式保存
        # 获取单个批次的数据
        X_data = list(batch_X.values())[0]
        Y_data = list(batch_Y.values())[0]
        
        # 检查X数据是否包含复数
        if np.iscomplexobj(X_data):
            # 创建一个2行的数组，第一行存实部，第二行存虚部
            X_separated = np.zeros((2, X_data.size), dtype=float)
            X_separated[0, :] = np.real(X_data)  # 第一行存实部
            X_separated[1, :] = np.imag(X_data)  # 第二行存虚部
            
            # 保存数据
            np.savetxt(f'{feature_path}.csv', X_separated, delimiter=',')
        else:
            # 非复数数据，直接保存
            np.savetxt(f'{feature_path}.csv', np.array([X_data]), delimiter=',')
        
        # 对Y数据执行相同的操作
        if np.iscomplexobj(Y_data):
            # 创建一个2行的数组，第一行存实部，第二行存虚部
            Y_separated = np.zeros((2, Y_data.size), dtype=float)
            Y_separated[0, :] = np.real(Y_data)  # 第一行存实部
            Y_separated[1, :] = np.imag(Y_data)  # 第二行存虚部
            
            # 保存数据
            np.savetxt(f'{label_path}.csv', Y_separated, delimiter=',')
        else:
            # 非复数数据，直接保存
            np.savetxt(f'{label_path}.csv', np.array([Y_data]), delimiter=',')


# 合并CSV文件的函数（如果需要）
def merge_csv_files(root_dir, source_dir, output_name):
    """合并目录中的所有CSV文件"""
    
    csv_list = glob.glob(os.path.join(source_dir, '*.csv'))
    print(f'发现{len(csv_list)}个CSV文件，开始合并...')
    
    # 确保输出文件不存在
    output_file = os.path.join(root_dir, f'{output_name}.csv')
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 按批次序号排序文件
    csv_list.sort()
    
    # 合并文件
    with open(output_file, 'wb') as outfile:
        for csv_file in tqdm(csv_list):
            with open(csv_file, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f'文件已成功合并到: {output_file}')
    
    # 可选：删除原始目录
    # shutil.rmtree(source_dir)
    
if __name__ == "__main__":
    generate_train_data(num_batch=4, root_dir="data/", dataset_type="train", num_bits=2048, poly='CRC-16', m_psk=4, n_fft=64,
                        n_cp=16, sample_rate=150, snr_db=50, wind_speed=1.5, label="bits", input="normalized_signal", use_hdf5=False, use_multiprocessing=False)