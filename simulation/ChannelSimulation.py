import numpy as np
from scipy.stats import expon

def gaussian_noise(signal, snr_db=20):
    """
    模拟信道传输，添加高斯白噪声
    
    参数:
    signal (numpy array): 输入信号，可以是复数信号
    snr_db (float): 信噪比，单位为dB，默认为20dB
    
    返回:
    numpy array: 噪声
    """
    snr = 10**(snr_db/10)
    signal_power = np.mean(np.abs(signal)**2)
    
    noise_power = signal_power / snr
    
    noise = np.sqrt(noise_power/2) * np.random.randn(len(signal))
    
    return noise


def alpha_dist_noise(input_array, alpha, beta, gamma, a, msnr):
    """
    生成alpha稳态分布噪声
    
    参数:
    input_array: 输入数组（可以是实数或复数）
    alpha: 特征参数，0 < alpha <= 2，在浅海条件下
    beta: 对称参数，-1 <= beta <= 1
    gamma: 比例参数，gamma > 0
    a: 位置参数
    msnr: 信噪比(dB)
    
    返回:
    output_array: alpha稳态噪声
    """
    if np.sum(np.imag(input_array)) == 0:
        comp_flag = 1 
    else:
        comp_flag = 2 
    
    array_len = len(input_array)
    
    k_alpha = alpha - 1 + np.sign(1 - alpha)
    
    if alpha != 1:
        beta2 = 2 * np.arctan(beta * np.tan(np.pi * alpha / 2)) / (np.pi * k_alpha)
        gamma2 = gamma * (1 + beta**2 * (np.tan(np.pi * alpha / 2)**2))**(1 / (2 * alpha))
    else:
        beta2 = beta
        gamma2 = 2 * gamma / np.pi
    
    w = expon.rvs(scale=1, size=array_len)
    
    gamma0 = -np.pi / 2 * beta2 * k_alpha / alpha
    
    gamma1 = np.zeros((comp_flag, array_len))
    for j in range(comp_flag):
        gamma1[j, :] = -np.pi/2 + np.pi * np.random.rand(array_len)
    
    x = np.zeros((comp_flag, array_len))
    
    if alpha != 1:
        for ik in range(array_len):
            for jk in range(comp_flag):
                term1 = np.sin(alpha * (gamma1[jk, ik] - gamma0))
                term2 = (np.cos(gamma1[jk, ik]))**(1/alpha)
                term3 = np.cos(gamma1[jk, ik] - alpha * (gamma1[jk, ik] - gamma0))
                term4 = (w[ik])**((1-alpha)/alpha)
                x[jk, ik] = (term1 / term2) * (term3 / term4)
    else:
        for ik in range(array_len):
            for jk in range(comp_flag):
                term1 = (np.pi/2 + beta2 * gamma1[jk, ik]) * np.tan(gamma1[jk, ik])
                term2 = beta2 * np.log10(w[ik] * np.cos(gamma1[jk, ik]) / (np.pi/2 + beta * gamma1[jk, ik]))
                x[jk, ik] = term1 - term2
    
    y = gamma2 * x
    
    if alpha != 1:
        u = y + a
    else:
        u = y + a + 2/np.pi * gamma * beta * np.log(2*gamma/np.pi)
    
    amp_lvl = np.sqrt(10**(msnr/10) * gamma)
    
    if comp_flag == 1:
        output_array = u[0, :] / amp_lvl
    else:
        output_array = (u[0, :] + 1j * u[1, :]) / amp_lvl
    
    return output_array

