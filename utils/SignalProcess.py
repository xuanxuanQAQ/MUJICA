import numpy as np


def sliding_window(signal, window_size, overlap_percent=50):
    """
    使用滑动窗口技术将复数信号分割成重叠窗口。
    
    参数:
        signal (np.ndarray): 输入信号，可以是以下形式:
                            - 复数数组 (n_samples,) 或 (n_samples, n_features)
                            - 实部和虚部分离的数组 (n_samples, 2) 或 (n_samples, n_features, 2)
        window_size (int): 每个窗口的样本数
        overlap_percent (float): 重叠百分比，默认50%
    
    返回:
        np.ndarray: 窗口化后的信号，形状为:
                    - 对于复数输入: (n_windows, window_size, n_features) 复数数组
                    - 对于分离实虚部输入: (n_windows, window_size, n_features, 2)
    """
    is_complex = is_complex_before = np.iscomplexobj(signal)
    
    if is_complex:
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
    else:
        if len(signal.shape) == 2 and signal.shape[1] == 2:
            signal = signal[:, 0] + 1j * signal[:, 1]
            signal = signal.reshape(-1, 1)
            is_complex = True
        elif len(signal.shape) == 3 and signal.shape[2] == 2:
            signal = signal[:, :, 0] + 1j * signal[:, :, 1]
            is_complex = True
    
    n_samples, n_features = signal.shape
    
    step = int(window_size * (1 - overlap_percent/100))
    step = max(1, step) 

    n_windows = (n_samples - window_size) // step + 1
    
    if is_complex:
        windows = np.zeros((n_windows, window_size, n_features), dtype=complex)
    else:
        windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        windows[i] = signal[start_idx:end_idx]
    
    if is_complex and not is_complex_before:
        real_part = np.real(windows)
        imag_part = np.imag(windows)
        windows_separate = np.concatenate((real_part, imag_part), axis=-1)
        return windows_separate
    
    if n_features == 1 and len(signal.shape) == 2:
        return windows.reshape(n_windows, window_size)
    
    return windows

def complex_to_channels(complex_signal):
    """
    将复数信号转换为双通道表示，分离实部和虚部。
    
    参数:
        complex_signal (np.ndarray): 输入的复数信号，可以是以下形状:
                                    - 一维数组 (n_samples,)
                                    - 二维数组 (n_samples, n_features)
                                    
    返回:
        np.ndarray: 转换后的双通道实数数组，形状为:
                    - 对于一维输入: (n_samples, 2)
                    - 对于二维输入: (n_samples, n_features, 2)
                    其中最后一维度的索引0是实部，索引1是虚部
    """
    if not np.iscomplexobj(complex_signal):
        raise ValueError("输入必须是复数数组")
    
    if len(complex_signal.shape) == 1:
        real_part = np.real(complex_signal)
        imag_part = np.imag(complex_signal)
        return np.stack((real_part, imag_part), axis=1)
    
    elif len(complex_signal.shape) == 2:
        real_part = np.real(complex_signal)
        imag_part = np.imag(complex_signal)
        return np.stack((real_part, imag_part), axis=2)
    
    else:
        raise ValueError("仅支持一维或二维复数数组")

def channels_to_complex(channel_signal):
    """
    将双通道表示（实部和虚部）转回复数信号。
    
    参数:
        channel_signal (np.ndarray): 输入的双通道实数信号，可以是以下形状:
                                    - (n_samples, 2) 其中[:,0]是实部，[:,1]是虚部
                                    - (n_samples, n_features, 2)
    
    返回:
        np.ndarray: 转换后的复数数组，形状为:
                    - 对于 (n_samples, 2) 输入: (n_samples,)
                    - 对于 (n_samples, n_features, 2) 输入: (n_samples, n_features)
    """
    if not (len(channel_signal.shape) >= 2 and channel_signal.shape[-1] == 2):
        raise ValueError("输入的最后一个维度必须是2（表示实部和虚部）")
    
    if len(channel_signal.shape) == 2:
        return channel_signal[:, 0] + 1j * channel_signal[:, 1]
    
    elif len(channel_signal.shape) == 3:
        return channel_signal[:, :, 0] + 1j * channel_signal[:, :, 1]
    
    else:
        raise ValueError("仅支持(n_samples, 2)或(n_samples, n_features, 2)形状的输入")