import numpy as np


def binary_to_string(binary_array):
    """
    将二进制数组转换为字符串格式
    
    参数:
    binary_array (numpy.ndarray): 包含0和1的二进制数组
    
    返回:
    str: 二进制字符串表示
    """
    return ''.join(str(bit) for bit in binary_array)

def signal_normalize(recieved_signal):
    """
    将信号幅度归一化到微幅波的10um级
    """
    max_value = np.max(np.abs(recieved_signal))
    scale_factor = 0.01 / max_value
    normalized_signal = recieved_signal * scale_factor
    
    return normalized_signal