import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import simulation
import modulation
import radar
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
import csv
import plot

def generate_batch_data(num_bits=2048, poly='CRC-16', m_psk=4, n_fft=64, comb_num=8, n_cp=16, 
                        sample_rate=200, fb=200, snr_db=20, wind_speed=1.5, label=None, input=None):
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
    ofdm_signal, frame_structure, _ = modulation.ofdm_modulation(complex_symbols, n_fft, n_cp, fb, pilot_pattern='comb', comb_num=comb_num)
    real_ofdm_signal = np.real(ofdm_signal)
    if label == 'ofdm_signal' or (isinstance(label, list) and 'ofdm_signal' in label):
        sample_labels['ofdm_signal'] = real_ofdm_signal
    interpolated_ofdm, time_indices = utils.interpolate_1d_timeseries(real_ofdm_signal, fb, sample_rate, method='linear')
    
    # 信道模拟
    # alpha_dist_noise = simulation.alpha_dist_noise(interpolated_ofdm, 1.5, 0, 1, 0, snr_db)
    gaussain_noise = simulation.gaussian_noise(interpolated_ofdm, snr_db)
    noised_signal = gaussain_noise + interpolated_ofdm
    
    # 微幅波传导后时域谱（input1-normalised_noised_signal
    normalised_noised_signal = utils.signal_normalize(noised_signal, target_amplitude=5*1e-5)
    if input == 'normalised_noised_signal' or (isinstance(input, list) and 'normalised_noised_signal' in input):
        sample_inputs['normalised_noised_signal'] = normalised_noised_signal
    
    # 添加有限振幅波
    duration = np.max(time_indices)
    dt = time_indices[1] - time_indices[0]
    # w, S = simulation.PM(wind_speed)
    # t, eta = simulation.generate_time_series(w, S, duration, dt)
    htt, V, M_scale, P_scale, cover_size, second, Num, patch_size, mesh_size, deta_x = simulation.PM3D(wind_speed, P_scale=4, M_scale=4, second=duration, fps=100, use_gpu=True)
    interpolated_htt, target_times = simulation.interpolate_htt(htt, Num, sample_rate, second, True)
    # 加入毛细波（待定）
    
    recieved_signal, center_signal, center_wave = simulation.add_signal_to_center_points(interpolated_htt, normalised_noised_signal)
    if label == 'ocean_wave' or (isinstance(label, list) and 'ocean_wave' in label):
        sample_labels['ocean_wave'] = center_wave
    if input == 'recieved_signal' or (isinstance(input, list) and 'recieved_signal' in input):
        sample_inputs['recieved_signal'] = center_signal
    
    ele_received, _, distance_resolu, f = simulation.ocean_surface_radar_backscattering(recieved_signal, wind_speed, 0, patch_size, mesh_size, deta_x, sample_rate)
    if input == 'ele_received' or (isinstance(input, list) and 'ele_received' in input):
        sample_inputs['ele_received'] = ele_received
        
    _, max_range_idx = radar.find_max_energy_range_bin(ele_received.T)
    phase_range, maxloc = radar.extract_phase_from_max_range_bin(ele_received.T, max_range_idx, range_search=10)
    unwrapped_phase, _ = radar.extract_and_unwrap_phase(ele_received, maxloc, method='compensated', filter_noise=False, distance_resolu=distance_resolu, f=f)
    
    if input == 'ele_received_phase' or (isinstance(input, list) and 'ele_received_phase' in input):
        sample_inputs['ele_received_phase'] = unwrapped_phase  # 存储相位（弧度）

    # # OFDM解调
    # ofdm_freq_symbols, symbol_mapping = demodulation.ofdm_preprocessing(noised_signal, frame_structure)
    # if input == 'ofdm_freq_symbols' or (isinstance(input, list) and 'ofdm_freq_symbols' in input):
    #     sample_inputs['ofdm_freq_symbols'] = ofdm_freq_symbols.flatten()
        
    # n_fft = frame_structure['n_fft']
    # channel_estimates = demodulation.estimate_channel(ofdm_freq_symbols, symbol_mapping, n_fft)
    # demodulated_symbols = demodulation.apply_equalization(ofdm_freq_symbols, channel_estimates, symbol_mapping, 'zf')
    
    # # QPSK解调（如果需要可以添加到输出）fs, f, M_psk, M_mod, mSig
    # demodulated_bits, decoded_symbols = demodulation.mpsk_demodulation(demodulated_symbols, m_psk)
    
    return sample_labels, sample_inputs, frame_structure
    
def get_dataset_paths(root_dir, dataset_type, label, input):
    """
    创建数据集存储路径和目录。
    支持label和input为字符串或字符串列表。
    
    参数:
    root_dir: 根目录
    dataset_type: 数据集类型 ('train', 'validation', 'test')
    label: 标签类型 - 字符串或字符串列表 (例如 'bits' 或 ['bits', 'crc_bits'])
    input: 输入类型 - 字符串或字符串列表 (例如 'normalized_signal' 或 ['normalized_signal', 'recieved_signal'])
    
    返回:
    包含路径信息的字典
    """
    # 确定数据集类型前缀和分母值
    if dataset_type == 'train':
        prefix = 'Train'
    elif dataset_type == 'val':
        prefix = 'Val'
    else:  # test
        prefix = 'Test'
    
    # 处理input和label，将它们转换为&连接的字符串
    # 如果已经是字符串，则直接使用；如果是列表，则用&连接
    if isinstance(input, list):
        input_str = '&'.join(input)
    else:
        input_str = input
        
    if isinstance(label, list):
        label_str = '&'.join(label)
    else:
        label_str = label
    
    # 构建目录名
    feature_dir = os.path.join(root_dir, f'{prefix}_{input_str}')
    label_dir = os.path.join(root_dir, f'{prefix}_{label_str}')
    
    # 确保目录存在
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # 生成文件前缀
    feature_prefix = f'{prefix}_{input_str}'
    label_prefix = f'{prefix}_{label_str}'
    
    return {
        'feature_dir': feature_dir,
        'label_dir': label_dir,
        'feature_prefix': feature_prefix,
        'label_prefix': label_prefix,
    }
    
def generate_train_data(num_batch=1, root_dir="data/", dataset_type="train", num_bits=2048, poly='CRC-16', m_psk=4, n_fft=64, comb_num=8,
                        n_cp=16, sample_rate=400, fb=200, snr_db=50, wind_speed=6, label="bits", input="normalized_signal", use_hdf5=False, use_multiprocessing=False):
    """
    生成训练、验证或测试数据
    
    参数:
    - num_batch: 批次总数
    - comb_num: 一个symbol内导频间隔
    - root_dir: 存储数据集的根目录
    - dataset_type: 数据集类型（train/val/test）
    - use_hdf5: 是否使用HDF5格式保存（与CSV相比更高效）
    - use_multiprocessing: 是否使用多进程加速数据生成
    
    返回:
    - 无返回值，数据保存到指定目录
    """
    # 验证数据集类型
    valid_types = ['train', 'val', 'test']
    if dataset_type not in valid_types:
        raise ValueError(f'dataset_type必须是{valid_types}之一')
    
    # 获取数据集存储路径信息
    paths = get_dataset_paths(root_dir, dataset_type, label, input)
    
    # 检查现有文件，确定开始索引
    feature_dir = paths['feature_dir']
    start_batch_idx = 0
    
    # 创建目录（如果不存在）
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(paths['label_dir'], exist_ok=True)
    
    # 确定当前已有的最大批次索引
    existing_files = []
    extension = '.h5' if use_hdf5 else '.csv'
    
    for filename in os.listdir(feature_dir):
        if filename.endswith(extension):
            try:
                # 从文件名中提取索引，例如 "0001.csv" -> 1
                idx = int(filename.split('.')[0])
                existing_files.append(idx)
            except ValueError:
                # 忽略不符合命名规则的文件
                continue
    
    if existing_files:
        start_batch_idx = max(existing_files) + 1
        print(f"发现现有数据文件，将从批次索引 {start_batch_idx} 开始继续生成")
    
    # 确定实际需要生成的批次数
    remaining_batches = num_batch - start_batch_idx
    if remaining_batches <= 0:
        print(f"已存在 {start_batch_idx} 个批次，指定生成 {num_batch} 个批次，无需额外生成")
        return
    
    print(f"将生成 {remaining_batches} 个新批次（从索引 {start_batch_idx} 到 {num_batch-1}）")
    
    start_time = time.time()
    
    if use_multiprocessing and remaining_batches > 1:
        # 使用多进程并行生成数据
        partial_gen = functools.partial(
            generate_batch_data,
            num_bits=num_bits,
            poly=poly,
            m_psk=m_psk,
            n_fft=n_fft,
            comb_num=comb_num,
            n_cp=n_cp,
            sample_rate=sample_rate,
            fb=fb,
            snr_db=snr_db,
            wind_speed=wind_speed,
            label=label,
            input=input
        )
        
        # 创建批次索引列表（从start_batch_idx开始）
        batch_indices = list(range(start_batch_idx, num_batch))
        
        # 使用ProcessPoolExecutor并行处理
        with ProcessPoolExecutor() as executor:
            with tqdm(total=remaining_batches, desc="生成数据批次") as pbar:
                for batch_idx, (batch_Y, batch_X, frame_structure) in zip(
                    batch_indices,
                    executor.map(partial_gen, batch_indices)
                ):
                    save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5)
                    pbar.update(1)
    else:
        # 顺序生成数据
        for batch_idx in tqdm(range(start_batch_idx, num_batch), desc="生成数据批次"):
            batch_Y, batch_X, frame_structure = generate_batch_data(
                num_bits=num_bits,
                poly=poly,
                m_psk=m_psk,
                n_fft=n_fft,
                comb_num=comb_num,
                n_cp=n_cp,
                sample_rate=sample_rate,
                fb=fb,
                snr_db=snr_db,
                wind_speed=wind_speed,
                label=label,
                input=input
            )
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
    对于字典类型的数据，将每一项单独保存到一行或一个数据集中
    对于复数数据，将实部和虚部分别处理
    
    参数:
    batch_idx: 批次索引
    batch_Y: 单个批次的标签数据（字典类型）
    batch_X: 单个批次的特征数据（字典类型）
    paths: 保存路径字典
    use_hdf5: 是否使用HDF5格式保存
    """
    feature_path = os.path.join(paths['feature_dir'], f'{batch_idx:04d}')
    label_path = os.path.join(paths['label_dir'], f'{batch_idx:04d}')
    
    if use_hdf5:
        # 使用HDF5格式保存
        with h5py.File(f'{feature_path}.h5', 'w') as f:
            # 遍历字典中的每一项
            for key, data in batch_X.items():
                # 创建组以存储每个键的数据
                group = f.create_group(key)
                
                # 检查是否包含复数，如果是则分离实部和虚部
                if np.iscomplexobj(data):
                    group.create_dataset('real', data=np.real(data))
                    group.create_dataset('imag', data=np.imag(data))
                else:
                    group.create_dataset('data', data=data)
        
        with h5py.File(f'{label_path}.h5', 'w') as f:
            # 遍历字典中的每一项
            for key, data in batch_Y.items():
                # 创建组以存储每个键的数据
                group = f.create_group(key)
                
                # 检查是否包含复数，如果是则分离实部和虚部
                if np.iscomplexobj(data):
                    group.create_dataset('real', data=np.real(data))
                    group.create_dataset('imag', data=np.imag(data))
                else:
                    group.create_dataset('data', data=data)
    else:
        # 处理CSV格式保存
        # 为特征数据创建CSV文件
        with open(f'{feature_path}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            
            # 写入标题行（包含所有键和数据类型信息）
            header = []
            for key in batch_X.keys():
                header.append(key)
                # 添加数据类型标记
                if np.iscomplexobj(batch_X[key]):
                    header.append(f"{key}_complex")
            writer.writerow(header)
            
            # 确定最大行数（如果数据长度不一样）
            max_rows = max([len(data) if hasattr(data, '__len__') else 1 for data in batch_X.values()])
            
            # 写入数据行
            for row_idx in range(max_rows):
                row_data = []
                for key, data in batch_X.items():
                    # 确保数据是可迭代的
                    if not hasattr(data, '__len__'):
                        data = [data]
                    
                    # 如果行索引超出数据范围，使用None
                    if row_idx >= len(data):
                        if np.iscomplexobj(data):
                            row_data.extend([None, None])  # 对于复数，添加两个None
                        else:
                            row_data.append(None)
                    else:
                        # 对于复数数据，分别添加实部和虚部
                        if np.iscomplexobj(data):
                            row_data.append(np.real(data[row_idx]))
                            row_data.append(np.imag(data[row_idx]))
                        else:
                            row_data.append(data[row_idx])
                
                writer.writerow(row_data)
        
        # 为标签数据创建CSV文件
        with open(f'{label_path}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            
            # 写入标题行（包含所有键和数据类型信息）
            header = []
            for key in batch_Y.keys():
                header.append(key)
                # 添加数据类型标记
                if np.iscomplexobj(batch_Y[key]):
                    header.append(f"{key}_complex")
            writer.writerow(header)
            
            # 确定最大行数（如果数据长度不一样）
            max_rows = max([len(data) if hasattr(data, '__len__') else 1 for data in batch_Y.values()])
            
            # 写入数据行
            for row_idx in range(max_rows):
                row_data = []
                for key, data in batch_Y.items():
                    # 确保数据是可迭代的
                    if not hasattr(data, '__len__'):
                        data = [data]
                    
                    # 如果行索引超出数据范围，使用None
                    if row_idx >= len(data):
                        if np.iscomplexobj(data):
                            row_data.extend([None, None])  # 对于复数，添加两个None
                        else:
                            row_data.append(None)
                    else:
                        # 对于复数数据，分别添加实部和虚部
                        if np.iscomplexobj(data):
                            row_data.append(np.real(data[row_idx]))
                            row_data.append(np.imag(data[row_idx]))
                        else:
                            row_data.append(data[row_idx])
                
                writer.writerow(row_data)


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
                        n_cp=16, sample_rate=400, fb=200, snr_db=50, wind_speed=1.5, label="bits", input="normalized_signal", use_hdf5=False, use_multiprocessing=False)