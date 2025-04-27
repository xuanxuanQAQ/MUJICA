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