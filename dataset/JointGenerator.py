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
import pandas as pd
import demodulation
import glob
import shutil
import json
import deeplearning as dl


def generate_data(input, output, sample_inputs, frame_structure, methods='custom', model=None):
    """
    生成批量数据并进行信道估计
    
    参数:
    input: 输入数据类型
    sample_inputs: 输入数据
    output: 输出数据标签
    frame_structure: 帧结构参数字典，包含num_ofdm_symbols和n_fft等参数
    methods: 使用的方法，可选'custom'或'dl'，默认为'custom'
    model: 深度学习模型，当methods为'dl'时使用，默认为None
    
    返回:
    sample_outputs: 包含处理结果的字典，其中包含信道估计等数据
    """
    
    sample_outputs = {}
    
    symbol_mapping = frame_structure['symbol_mapping']
    symbol_mapping_array = [symbol_mapping[i] for i in sorted(symbol_mapping.keys())]
    symbol_mapping_array = np.array(symbol_mapping_array)
    
    if input == 'ofdm_freq_symbols' and output =='channel_estimates':
        channel_estimates_list = []
        
        for single_input in tqdm(sample_inputs, desc='Processing inputs'):
            if methods == 'dl':
                channel_estimates = dl.predict(model, single_input, symbol_mapping_array, 
                                                frame_structure['num_ofdm_symbols'][0] * frame_structure['n_fft'][0], 
                                                'model/best_model.pth')
                channel_estimates = np.squeeze(channel_estimates, axis=0)
                channel_estimates = channel_estimates[:, 0] + 1j * channel_estimates[:, 1]
                single_channel_est = channel_estimates.numpy()
            else:
                single_input = single_input.reshape(frame_structure['num_ofdm_symbols'][0], frame_structure['n_fft'][0])
                single_channel_est = demodulation.estimate_channel(single_input, symbol_mapping, frame_structure['n_fft'][0])
                single_channel_est = single_channel_est.flatten()
                
            channel_estimates_list.append(single_channel_est)
            
        channel_estimates = np.array(channel_estimates_list)
        sample_outputs['channel_estimates'] = channel_estimates
    
    return sample_outputs



def get_dataset_paths(root_dir, dataset_type, num_batch, input):
    """
    创建数据集存储路径和目录
    
    参数:
    root_dir: 根目录
    dataset_type: 数据集类型 ('train', 'validation', 'test')
    num_batch: 批次数量
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
    
    suffix = f'B{num_batch//denom:02d}'
    feature_dir = os.path.join(root_dir, f'{prefix}_{input}_{suffix}')
    os.makedirs(feature_dir, exist_ok=True)
    feature_prefix = f'{prefix}_{input}_{suffix}'
    
    return {
        'feature_dir': feature_dir,
        'feature_prefix': feature_prefix,
    }

    
def generate_joint_data(num_batch, root_dir="data/", dataset_type="train", input="ofdm_freq_symbols", output="channel_estimates", methods='custom', model=None,use_hdf5=False, use_multiprocessing=False):
    """
    生成训练、验证或测试数据
    
    参数:
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
    
    tmp_paths = get_dataset_paths(root_dir, dataset_type, num_batch, input)
    input_path = tmp_paths['feature_dir']
    files = []
    for ext in ['*.csv', '*.h5']:
        files.extend(glob.glob(os.path.join(input_path, ext)))
    files.sort()
    
    paths = get_dataset_paths(root_dir, dataset_type, num_batch, output)
    
    batch_X = []
    for file in files:
        if file.endswith('.h5'):
            with h5py.File(file, 'r') as f:
                if 'real' in f and 'imag' in f:
                    data = f['real'][()] + 1j * f['imag'][()]
                else:
                    data = f['data'][()]
                batch_X.append(data)
        else:
            data = np.loadtxt(file, delimiter=',')
            if data.shape[0] == 2:
                data = data[0, :] + 1j * data[1, :]
            batch_X.append(data)
    
    batch_X = np.array(batch_X)

    structure_dir = os.path.join(root_dir, f'{dataset_type}_frame_structure.json')
    structure_config = pd.read_json(structure_dir)
    frame_structure = structure_config.to_dict()

    start_time = time.time()
    
    if use_multiprocessing and num_batch > 1:
        partial_gen = functools.partial(
            generate_data,
            input=input,
            output=output,
            sample_inputs=batch_X,
            frame_structure=frame_structure,
            methods='custom',
            model=None
        )
        
        # 使用ProcessPoolExecutor并行处理
        with ProcessPoolExecutor() as executor:
            with tqdm(total=num_batch) as pbar:
                for batch_idx, batch_Y in enumerate(
                    executor.map(partial_gen, range(num_batch))
                ):
                    save_batch(batch_idx, batch_Y, paths, use_hdf5)
                    pbar.update(1)
    else:
        # 顺序生成数据
        batch_Y = generate_data(input, output, batch_X, frame_structure, methods, model)
        for batch_idx in tqdm(range(num_batch)):
            save_batch(batch_idx, batch_Y[output][batch_idx], paths, use_hdf5)
    
    # Save frame_structure as JSON
    if num_batch > 0:  # Only save if we generated at least one batch
        frame_structure_path = os.path.join(root_dir, f'{dataset_type}_frame_structure.json')
        with open(frame_structure_path, 'w') as f:
            json.dump(frame_structure, f, indent=4)
    
    end_time = time.time()
    print(f'联合生成：生成并保存{num_batch}个批次耗时: {end_time-start_time:.2f}秒')
    print(f'数据已保存到目录: {os.path.abspath(root_dir)}')
    
    
def save_batch(batch_idx, batch_Y, paths, use_hdf5=False):
    """
    保存单个批次的数据，支持CSV或HDF5格式
    对于复数数据，将实部和虚部分别存储在CSV的相邻两行中
    一行存储实部，下一行存储虚部，每两行表示一个完整的数据
    
    参数:
    batch_idx: 批次索引
    batch_Y: 单个批次的标签数据
    paths: 保存路径字典
    use_hdf5: 是否使用HDF5格式保存
    """
    feature_path = os.path.join(paths['feature_dir'], f'{batch_idx:04d}')
    
    if use_hdf5:
        with h5py.File(f'{feature_path}.h5', 'w') as f:
            # 直接处理numpy数组
            if np.iscomplexobj(batch_Y):
                f.create_dataset('real', data=np.real(batch_Y))
                f.create_dataset('imag', data=np.imag(batch_Y))
            else:
                f.create_dataset('data', data=batch_Y)
    else:
        # 处理CSV格式保存
        Y_data = batch_Y
        
        if np.iscomplexobj(Y_data):
            # 创建一个2行的数组，第一行存实部，第二行存虚部
            Y_separated = np.zeros((2, Y_data.size), dtype=float)
            Y_separated[0, :] = np.real(Y_data)  # 第一行存实部
            Y_separated[1, :] = np.imag(Y_data)  # 第二行存虚部
            
            # 保存数据
            np.savetxt(f'{feature_path}.csv', Y_separated, delimiter=',')
        else:
            # 非复数数据，直接保存
            np.savetxt(f'{feature_path}.csv', np.array([Y_data]), delimiter=',')
