

def binary_to_string(binary_array):
    """
    将二进制数组转换为字符串格式
    
    参数:
    binary_array (numpy.ndarray): 包含0和1的二进制数组
    
    返回:
    str: 二进制字符串表示
    """
    return ''.join(str(bit) for bit in binary_array)