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

def signal_normalize(received_signal, target_amplitude=1e-7):
    """
    将信号幅度归一化到指定目标幅度
    
    Args:
        received_signal: 输入信号
        target_amplitude: 目标最大幅度，默认100nm
    
    Returns:
        归一化后的信号
    """
    max_value = np.max(np.abs(received_signal))
    
    # 避免除零错误
    if max_value == 0:
        return received_signal
    
    scale_factor = target_amplitude / max_value
    normalized_signal = received_signal * scale_factor
    
    return normalized_signal