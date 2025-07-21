import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class STDCTLayer(nn.Module):
    """PyTorch STDCT层，支持可微分操作"""
    
    def __init__(self, n_fft=512, hop_length=None, win_length=None,
                 window='hann', center=True, trainable=False):
        super(STDCTLayer, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.center = center
        
        # 创建DCT变换矩阵
        self.register_buffer('dct_matrix', self._create_dct_matrix())
        
        # 创建窗函数
        if window == 'hann':
            window_func = torch.hann_window(self.win_length)
        elif window == 'hamming':
            window_func = torch.hamming_window(self.win_length)
        elif window == 'blackman':
            window_func = torch.blackman_window(self.win_length)
        else:
            window_func = torch.ones(self.win_length)
        
        # 调整窗函数长度
        if self.win_length < self.n_fft:
            pad_amount = (self.n_fft - self.win_length) // 2
            window_func = F.pad(window_func, 
                              (pad_amount, self.n_fft - self.win_length - pad_amount))
        
        if trainable:
            self.window = nn.Parameter(window_func)
        else:
            self.register_buffer('window', window_func)
    
    def _create_dct_matrix(self):
        """创建DCT变换矩阵"""
        dct_matrix = torch.zeros(self.n_fft, self.n_fft)
        
        # DCT-II矩阵
        for k in range(self.n_fft):
            for n in range(self.n_fft):
                if k == 0:
                    dct_matrix[k, n] = math.sqrt(1.0 / self.n_fft)
                else:
                    dct_matrix[k, n] = math.sqrt(2.0 / self.n_fft) * \
                                     math.cos(math.pi * k * (n + 0.5) / self.n_fft)
        
        return dct_matrix
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入信号 (batch, length) 或 (batch, channels, length)
            
        返回:
            STDCT系数 (batch, channels, n_frames, n_fft)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)
        
        batch_size, channels, length = x.shape
        
        # 中心化填充
        if self.center:
            pad_amount = self.n_fft // 2
            x = F.pad(x, (pad_amount, pad_amount), mode='reflect')
            length += 2 * pad_amount
        
        # 分帧
        frames = self._frame_signal(x, length)  # (batch, channels, n_frames, n_fft)
        
        # 应用窗函数
        windowed_frames = frames * self.window.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # 应用DCT
        # windowed_frames: (batch, channels, n_frames, n_fft)
        # dct_matrix: (n_fft, n_fft)
        stdct_coeffs = torch.matmul(windowed_frames, self.dct_matrix.T)
        
        return stdct_coeffs
    
    def _frame_signal(self, x, length):
        """信号分帧"""
        batch_size, channels, _ = x.shape
        n_frames = 1 + (length - self.n_fft) // self.hop_length
        
        frames = torch.zeros(batch_size, channels, n_frames, self.n_fft, 
                           device=x.device, dtype=x.dtype)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            frames[:, :, i, :] = x[:, :, start:end]
        
        return frames
    
    def inverse(self, stdct_coeffs, target_length=None):
        """
        逆STDCT变换
        
        参数:
            stdct_coeffs: STDCT系数 (batch, channels, n_frames, n_fft)
            
        返回:
            重建信号 (batch, channels, length)
        """
        
        if stdct_coeffs.dim() == 3:
            stdct_coeffs = stdct_coeffs.unsqueeze(1)
              
        batch_size, channels, n_frames, n_fft = stdct_coeffs.shape
        
        # 应用逆DCT
        frames = torch.matmul(stdct_coeffs, self.dct_matrix)
        
        # 应用窗函数
        windowed_frames = frames * self.window.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # 重叠相加重建
        if self.center:
            signal_length = (n_frames - 1) * self.hop_length + n_fft
        else:
            signal_length = (n_frames - 1) * self.hop_length + n_fft
        
        signal = torch.zeros(batch_size, channels, signal_length, 
                           device=stdct_coeffs.device, dtype=stdct_coeffs.dtype)
        window_sum = torch.zeros(signal_length, device=stdct_coeffs.device, dtype=stdct_coeffs.dtype)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + n_fft
            signal[:, :, start:end] += windowed_frames[:, :, i, :]
            window_sum[start:end] += self.window
        
        # 归一化
        window_sum = torch.clamp(window_sum, min=1e-8)
        signal = signal / window_sum.unsqueeze(0).unsqueeze(0)
        
        if self.center:
            # 移除填充
            pad_amount = self.n_fft // 2
            signal = signal[:, :, pad_amount:-pad_amount]

        if target_length is not None:
            current_length = signal.shape[2] if signal.dim() == 3 else signal.shape[1]
            if current_length != target_length:
                if signal.dim() == 3:  # (batch, channels, length)
                    if current_length > target_length:
                        signal = signal[:, :, :target_length]
                    else:
                        pad_amount = target_length - current_length
                        signal = F.pad(signal, (0, pad_amount))
                else:  # (batch, length)
                    if current_length > target_length:
                        signal = signal[:, :target_length]
                    else:
                        pad_amount = target_length - current_length
                        signal = F.pad(signal, (0, pad_amount))
        
        return signal.squeeze(1) if signal.shape[1] == 1 else signal