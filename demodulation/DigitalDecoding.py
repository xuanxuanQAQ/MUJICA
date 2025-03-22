import numpy as np
from modulation.DigitalEncoding import crc_remainder


def crc_check(input_bits, polynomial_bits, check_value):
    """
    检查CRC是否正确
    
    参数:
    input_bits: 输入比特串，不包含CRC值
    polynomial_bits: 多项式比特串
    check_value: 要验证的CRC值
    
    返回:
    布尔值，表示CRC校验是否通过
    """
    remainder = crc_remainder(input_bits, polynomial_bits, '0')
    
    # 确保check_value和remainder格式一致
    if isinstance(check_value, str) and isinstance(remainder, np.ndarray):
        check_array = np.array([int(bit) for bit in check_value])
        return np.array_equal(remainder, check_array)
    elif isinstance(check_value, np.ndarray) and isinstance(remainder, np.ndarray):
        return np.array_equal(remainder, check_value)
    elif isinstance(remainder, np.ndarray):
        remainder_str = ''.join(str(bit) for bit in remainder)
        return remainder_str == check_value
    else:
        return remainder == check_value
