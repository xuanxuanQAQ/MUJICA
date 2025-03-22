import numpy as np
import os
import time
from tqdm import tqdm
import h5py
from concurrent.futures import ProcessPoolExecutor
import functools

def generate_batch_data(batch_idx, batch_size, H, start_channel, terminate_channel, SNRdb, mode, Pilotnum):
    """生成单个批次的数据"""
    input_labels = []
    input_samples = []
    for _ in range(batch_size):
        # 生成随机比特数据
        bits = [np.random.binomial(n=1, p=0.5, size=(128 * 4,)) for _ in range(4)]
        X = bits
        
        # 随机选择信道
        channel_idx = np.random.randint(start_channel, terminate_channel+1)
        HH = H[channel_idx]
        
        # 生成接收信号
        YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20
        
        # 构建标签
        XX = np.concatenate(bits, 0)
        
        input_labels.append(XX)
        input_samples.append(YY)
    
    return np.asarray(input_samples), np.asarray(input_labels)

def get_dataset_paths(root_dir, dataset_type, start_channel, terminate_channel, num_batch):
    """创建数据集存储路径和目录"""
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
    
    # 构建目录名
    suffix = f'C{start_channel:02d}{terminate_channel:02d}_{num_batch//denom:02d}'
    feature_dir = os.path.join(root_dir, f'{prefix}_signal_{suffix}')
    label_dir = os.path.join(root_dir, f'{prefix}_label_{suffix}')
    
    # 确保目录存在
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    return {
        'feature_dir': feature_dir,
        'label_dir': label_dir,
        'feature_prefix': f'{prefix}_signal_{suffix}',
        'label_prefix': f'{prefix}_label_{suffix}',
    }

def save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5=False):
    """保存单个批次的数据，支持CSV或HDF5格式"""
    feature_path = os.path.join(paths['feature_dir'], f'{batch_idx:04d}')
    label_path = os.path.join(paths['label_dir'], f'{batch_idx:04d}')
    
    if use_hdf5:
        # 使用HDF5格式保存（更高效）
        with h5py.File(f'{feature_path}.h5', 'w') as f:
            f.create_dataset('data', data=batch_Y, compression='gzip')
        
        with h5py.File(f'{label_path}.h5', 'w') as f:
            f.create_dataset('data', data=batch_X, compression='gzip')
    else:
        # 使用CSV格式保存（保持兼容性）
        np.savetxt(f'{feature_path}.csv', batch_Y, delimiter=',')
        np.savetxt(f'{label_path}.csv', batch_X, delimiter=',')

def generate_train_data(batch_size, num_batch, root_dir, dataset_type, start_channel, terminate_channel, 
                       H=None, SNRdb=25, mode=0, Pilotnum=8, use_hdf5=False, use_multiprocessing=False):
    """
    生成训练、验证或测试数据
    
    参数:
    - batch_size: 每个批次的样本数
    - num_batch: 批次总数
    - root_dir: 存储数据集的根目录
    - dataset_type: 数据集类型（train/validation/test）
    - start_channel: 开始信道索引（0-9000）
    - terminate_channel: 结束信道索引（0-9000）
    - H: 信道矩阵，如果为None，将从默认位置加载
    - SNRdb: 信噪比，单位dB
    - mode: 模式参数
    - Pilotnum: 导频数量
    - use_hdf5: 是否使用HDF5格式保存（与CSV相比更高效）
    - use_multiprocessing: 是否使用多进程加速数据生成
    
    返回:
    - 无返回值，数据保存到指定目录
    """
    # 验证数据集类型
    valid_types = ['train', 'validation', 'test']
    if dataset_type not in valid_types:
        raise ValueError(f'dataset_type必须是{valid_types}之一')
    
    # 如果未提供H，从默认位置加载
    if H is None:
        import scipy.io as scio
        data_load_address = './data'
        mat = scio.loadmat(data_load_address+'/Htrain.mat')
        x_train = mat['H_train']
        H = x_train[:,:,:,0] + 1j * x_train[:,:,:,1]
    
    # 获取数据集存储路径信息
    paths = get_dataset_paths(root_dir, dataset_type, start_channel, terminate_channel, num_batch)
    
    start_time = time.time()
    
    if use_multiprocessing and num_batch > 1:
        # 使用多进程并行生成数据
        partial_gen = functools.partial(
            generate_batch_data, 
            batch_size=batch_size,
            H=H,
            start_channel=start_channel,
            terminate_channel=terminate_channel,
            SNRdb=SNRdb,
            mode=mode,
            Pilotnum=Pilotnum
        )
        
        # 使用ProcessPoolExecutor并行处理
        with ProcessPoolExecutor() as executor:
            with tqdm(total=num_batch) as pbar:
                for batch_idx, (batch_Y, batch_X) in enumerate(
                    executor.map(partial_gen, range(num_batch))
                ):
                    save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5)
                    pbar.update(1)
    else:
        # 顺序生成数据
        for batch_idx in tqdm(range(num_batch)):
            batch_Y, batch_X = generate_batch_data(
                batch_idx, batch_size, H, start_channel, terminate_channel, SNRdb, mode, Pilotnum
            )
            save_batch(batch_idx, batch_Y, batch_X, paths, use_hdf5)
    
    end_time = time.time()
    print(f'生成并保存{num_batch}个批次（每批{batch_size}个样本）耗时: {end_time-start_time:.2f}秒')
    print(f'数据已保存到目录: {os.path.abspath(root_dir)}')

# 合并CSV文件的函数（如果需要）
def merge_csv_files(root_dir, source_dir, output_name):
    """合并目录中的所有CSV文件"""
    import glob
    import shutil
    
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