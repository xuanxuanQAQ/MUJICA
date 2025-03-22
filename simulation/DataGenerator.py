import numpy as np

def generate_random_binary(length):
    """
    生成指定长度的随机二进制序列
    
    参数:
    length (int): 要生成的二进制位数
    
    返回:
    numpy.ndarray: 包含0和1的随机二进制序列
    """
    # 使用numpy生成随机二进制序列
    binary_sequence = np.random.randint(0, 2, length)
    
    return binary_sequence

