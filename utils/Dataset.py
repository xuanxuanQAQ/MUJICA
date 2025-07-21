import numpy as np

def trim_input_with_label(processed_phases, label, raloc):
    # 获取原始数据长度
    start_idx = raloc
    end_idx = raloc + len(label)
    available_length = processed_phases.shape[1] - start_idx

    # 如果可用长度小于所需长度，进行零填充
    if available_length < len(label):
        # 创建一个全零数组，形状为 [通道数, 所需长度]
        padded_input = np.zeros((processed_phases.shape[0], len(label)))
        
        # 复制可用的数据
        padded_input[:, :available_length] = processed_phases[:, start_idx:processed_phases.shape[1]]
        
        # 使用零填充后的数据
        input = padded_input
    else:
        # 原始长度足够，直接截取
        input = processed_phases[:, start_idx:end_idx]
        
    return input

def majority_vote_decoder(rxDatas):
    """
    对多个接收器通道的BPSK解调数据进行多数投票解码
    
    参数:
    rxDatas : list
        包含多个通道解调数据的列表，每个元素是一个二进制数组
        
    返回值:
    numpy.ndarray
        多数投票后的解码结果
    """
    # 确定所有通道数据的最小长度
    min_length = min(len(rx) for rx in rxDatas)
    
    # 创建结果数组
    decoded_signal = np.zeros(min_length, dtype=int)
    
    # 对每个位置进行多数投票
    for i in range(min_length):
        # 收集每个通道在当前位置的比特值
        votes = [rx[i] for rx in rxDatas]
        
        # 计算0和1的出现次数
        count_zeros = votes.count(0)
        count_ones = votes.count(1)
        
        # 根据多数投票确定结果
        if count_ones > count_zeros:
            decoded_signal[i] = 1
        else:
            decoded_signal[i] = 0
    
    return decoded_signal

def load_data_from_csv(file_path, columns=None):
    """
    从CSV文件加载数据，支持选择特定列
    
    参数:
    file_path : str
        CSV文件的路径
    columns : str or list of str, optional
        要加载的列名。如果为None，则加载所有数值列
        
    返回值:
    numpy.ndarray
        加载的数据，形状为 [通道数, 时间步数]
    """
    try:
        import pandas as pd
        
        # 使用pandas读取CSV文件，保留列名
        df = pd.read_csv(file_path)
        
        # 如果指定了列名，只选择这些列
        if columns is not None:
            if isinstance(columns, str):
                # 单列名情况
                if columns in df.columns:
                    data = df[columns].values
                    # 确保数据是二维的
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                else:
                    raise ValueError(f"列名 '{columns}' 不存在于CSV文件中")
            else:
                # 多列名情况
                valid_columns = [col for col in columns if col in df.columns]
                if not valid_columns:
                    raise ValueError(f"指定的列名均不存在于CSV文件中")
                data = df[valid_columns].values
        else:
            # 如果没有指定列名，加载所有数值列
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                raise ValueError(f"CSV文件中没有数值列")
            data = numeric_df.values
        
        # 转置数据以匹配 [通道数, 时间步数] 的格式
        data = data.T
        
        return data
        
    except Exception as e:
        # 回退到基本的numpy加载方式
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            
            # 如果数据只有一列，确保它是二维的
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            # 转置数据以匹配 [通道数, 时间步数] 的格式
            data = data.T
            
            return data
        except Exception as nested_e:
            raise ValueError(f"无法加载文件 {file_path}: {str(e)}\n尝试使用numpy加载时出错: {str(nested_e)}")