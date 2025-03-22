import numpy as np

def crc_remainder(input_bits, polynomial_bitstring, initial_filler='0'):
    """
    计算CRC校验余数
    
    参数:
    input_bits: 输入比特，可以是NumPy数组或字符串
    polynomial_bitstring: 多项式名称或比特串，如 "CRC-16" 或 "1101"
    initial_filler: 初始填充值，通常为'0'或'1'
    
    返回:
    用于校验的CRC余数（NumPy数组格式）
    """
    # 常用的CRC多项式
    CRC_POLYNOMIALS = {
        'CRC-4-ITU': '11001',         # x^4 + x^3 + 1
        'CRC-8': '110011011',         # x^8 + x^7 + x^6 + x^4 + x^2 + 1
        'CRC-8-CCITT': '100110001',   # x^8 + x^6 + x^4 + x^3 + x + 1
        'CRC-16': '11000000000000101', # x^16 + x^15 + x^2 + 1
        'CRC-16-CCITT': '10001000000100001', # x^16 + x^12 + x^5 + 1
        'CRC-32': '100000100110000010001110110110111', # x^32 + x^26 + x^23 + ... + x^4 + x^2 + x + 1
    }

    # 获取多项式
    if polynomial_bitstring in CRC_POLYNOMIALS:
        poly_str = CRC_POLYNOMIALS[polynomial_bitstring]
    else:
        poly_str = polynomial_bitstring
    
    # 将多项式转换为NumPy数组
    polynomial = np.array([int(bit) for bit in poly_str])
    
    # 确保输入位是NumPy数组
    if isinstance(input_bits, str):
        input_array = np.array([int(bit) for bit in input_bits])
    else:
        input_array = input_bits.copy()
    
    len_input = len(input_array)
    len_polynomial = len(polynomial)
    
    # 初始填充 - 使用NumPy连接
    filler_array = np.full(len_polynomial - 1, int(initial_filler))
    padded_input = np.concatenate([input_array, filler_array])
    
    # 移位寄存器模拟
    for i in range(len_input):
        if padded_input[i] == 1:
            padded_input[i:i+len_polynomial] = np.bitwise_xor(
                padded_input[i:i+len_polynomial], 
                polynomial
            )
    
    # 获取余数
    remainder = padded_input[-(len_polynomial-1):]
    return remainder

def crc_check(input_bitstring, polynomial_bitstring, check_value):
    """
    检查CRC是否正确
    
    参数:
    input_bitstring: 输入比特串，不包含CRC值
    polynomial_bitstring: 多项式比特串
    check_value: 要验证的CRC值
    
    返回:
    布尔值，表示CRC校验是否通过
    """
    remainder = crc_remainder(input_bitstring, polynomial_bitstring, '0')
    return remainder == check_value

def add_crc(input_bits, polynomial_name):
    """
    添加CRC校验码
    
    参数:
    input_bits: 输入比特，可以是NumPy数组或字符串
    polynomial_name: CRC多项式名称，如 "CRC-16"
    
    返回:
    带有CRC校验码的比特序列（与输入相同类型）
    """
    # 计算CRC余数
    remainder = crc_remainder(input_bits, polynomial_name)
    
    # 根据输入类型确定输出类型
    if isinstance(input_bits, str):
        # 如果输入是字符串，将余数转换为字符串并连接
        remainder_str = ''.join(str(bit) for bit in remainder)
        return input_bits + remainder_str
    else:
        # 如果输入是NumPy数组，直接连接数组
        return np.concatenate([input_bits, remainder])


