import numpy as np
from scipy.fftpack import ifft2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import time
from scipy.interpolate import interp1d
from math import ceil

def PM3D(V=6, D=-60, cover_size=10, P_scale=8, M_scale=2, wind_damp=0.01, second = 5, fps=50, plot=False, use_gpu=True):
    """
    Generate a 3D ocean wave simulation using the Pierson-Moskowitz spectrum with GPU acceleration.
    
    Parameters:
    V (float): Wind speed in m/s.  
    D (float): Wind direction in degrees.
    cover_size (float): Cover length in meters.
    P_scale (float): Scale factor for the wave spectrum.
    M_scale (float): Scale factor for the mesh size.
    wind_damp (float): Attenuation factor for the waves.
    fps (float): Frames per second.
    plot (bool): Whether to plot the simulation.
    use_gpu (bool): Whether to use GPU acceleration.
    
    Returns:
    htt (numpy.ndarray): Height field of the ocean surface.
    """
    
    # 检查是否使用GPU
    if use_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.fftpack import ifft2 as gpu_ifft2
            xp = cp  # 使用cupy作为计算库
        except ImportError:
            print("未找到CuPy库，退回到CPU计算。请使用 'pip install cupy-cuda11x' 安装CuPy (根据您的CUDA版本调整)")
            use_gpu = False
            xp = np
            from scipy.fftpack import ifft2 as gpu_ifft2
    else:
        xp = np  # 使用numpy作为计算库
        from scipy.fftpack import ifft2 as gpu_ifft2

    # Parameters
    patch_size = 25.1 * P_scale  # 102m spatial range, determines kx resolution
    mesh_size = int(1024 * M_scale) # FFT size (128/256/384/512 for speed)
    g = 9.81  # Gravitational constant
    length = V * V * 0.9128  # Domain wave length
    wind_dir = D * np.pi / 180  # Wind direction in radians
    deta_k = 2 * np.pi / patch_size
    deta_x = patch_size / mesh_size
    A = 0.0081 * patch_size**2  # Amplitude
    B = 1 / 8 / np.pi / deta_x

    # Generate wave spectrum - 使用GPU
    nn, mm = np.meshgrid(np.arange(1, mesh_size + 1), np.arange(1, mesh_size + 1))
    if use_gpu:
        nn = xp.asarray(nn)
        mm = xp.asarray(mm)

    # Calculate wave vector - 使用GPU
    kx = (2 * np.pi * (mm - 1 - mesh_size / 2) / patch_size).astype(xp.float32)
    ky = (2 * np.pi * (nn - 1 - mesh_size / 2) / patch_size).astype(xp.float32)
    sign_correction = xp.mod(mm + nn - 2, 2).astype(bool)

    k = xp.sqrt(kx**2 + ky**2).astype(xp.float32)
    # Wind modulation
    w_dot_k = (kx / k * xp.cos(wind_dir) + ky / k * xp.sin(wind_dir)).astype(xp.float32)  # Projection of normalized wave number vector on wind direction

    # Spectrum at given point
    P = (A * xp.exp(-0.74 * g**2 / (k**2 * V**4)) * (w_dot_k**2)).astype(xp.float32) / 2.36  # Gravitational PM spectrum * cos(theta)
    wave_limit = patch_size / 100
    P = P * xp.exp(-k**2 * wave_limit**2) / k**3
    # Filter waves moving in the wrong direction
    P[w_dot_k < 0] = P[w_dot_k < 0] * wind_damp
    P[xp.isnan(P)] = 0

    # Calculate initial surface in frequency domain - 使用GPU
    # RANDN - GAUSSIAN | RAND - NORMAL
    H0 = mesh_size / xp.sqrt(2) * (xp.random.randn(mesh_size, mesh_size) + 1j * xp.random.randn(mesh_size, mesh_size)) * xp.sqrt(P)
    # Get mirrored value of initial surface
    # 注意: cupy不支持rot90，要改为等价操作
    if use_gpu:
        # 等价于 np.rot90(np.conj(H0), 2)
        Hm = xp.flip(xp.flip(xp.conj(H0), 0), 1)
    else:
        Hm = np.rot90(np.conj(H0), 2)  # Equivalent to flipud(fliplr(B))

    # Dispersion
    W = xp.sqrt(g * k)

    # Cover parameters
    cn = int(np.floor(mesh_size * cover_size / patch_size))  # Cover length /m
    vn = int(np.floor(0.5 * patch_size / cover_size))
    
    # 这些网格只用于绘图，保持为NumPy数组
    X, Y = np.meshgrid(np.arange(-cn/2, cn/2), np.arange(-cn/2, cn/2))

    ts = 1 / fps
    Num = int(1/ts)
    

    if plot:
        # Create figure for visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X*deta_x, Y*deta_x, np.zeros((cn, cn)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('H (m)')
        plt.colorbar(surf, ax=ax)
        ax.set_aspect('equal')

    # Create htt array with the correct size - 在CPU上
    total_frames = ceil(Num * second)
    htt = np.zeros((cn, cn, total_frames), dtype=np.float32)

    # Time loop
    for tn in range(1, total_frames + 1):
        time_val = 0 + tn * ts
        
        # Update according to the dispersion relation - 在GPU上
        Hkt = H0 * xp.exp(1j * W * time_val) + Hm * xp.exp(-1j * W * time_val)  # Ensure Hkt has conjugate symmetry
        
        # Generate HeightField at time t using ifft - 在GPU上
        Ht = B * xp.real(gpu_ifft2(Hkt))
        Ht[sign_correction] = -1.0 * Ht[sign_correction]
        
        # 提取子区域并转换回CPU (如果使用GPU)
        subregion = Ht[vn*cn:vn*cn+cn, vn*cn:vn*cn+cn]
        if use_gpu:
            subregion_cpu = cp.asnumpy(subregion)  # 将GPU数据转回CPU
        else:
            subregion_cpu = subregion
        
        if plot:
            # Update plot
            if tn % 10 == 0:  # Update plot every 10 frames for speed
                ax.clear()
                surf = ax.plot_surface(X*deta_x, Y*deta_x, subregion_cpu, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('H (m)')
                ax.set_title(f'Time: {time_val:.2f}s')
                plt.pause(0.01)
        
        # Store data - 在CPU上
        htt[:, :, tn-1] = subregion_cpu
        
        if plot:
            plt.show(block=False)
    
    # 清理GPU内存（如果使用）
    if use_gpu:
        # 释放不再需要的GPU内存
        del Hkt, Ht, H0, Hm, P, k, kx, ky, W, w_dot_k, sign_correction
        cp.get_default_memory_pool().free_all_blocks()
        
    return htt, V, M_scale, P_scale, cover_size, second, Num, patch_size, mesh_size, deta_x


def interpolate_htt(htt, original_fps, target_fps, original_duration, use_gpu=False):
    """
    高效地对海洋波浪模拟数据htt进行时间维度上的插值，支持GPU加速
    
    参数:
    htt (numpy.ndarray): 波浪高度场，形状为 [height, width, frames]
    original_fps (float): 原始帧率，等于1/Num
    target_dt (float): 目标时间步长
    original_duration (float): 原始数据的总时长(秒)
    use_gpu (bool): 是否使用GPU加速，默认为False
    
    返回:
    numpy.ndarray: 插值后的波浪高度场，形状为 [height, width, new_frames]
    numpy.ndarray: 新的时间索引数组
    """
    height, width, frames = htt.shape
    target_dt = 1/target_fps  # 目标时间步长
    # 如果启用GPU且可用
    if use_gpu:
        try:
            import cupy as cp
            
            # 将数据转移到GPU
            htt_gpu = cp.asarray(htt)
            
            original_times = cp.linspace(0, original_duration, frames)
            target_times = cp.arange(0, original_duration + target_dt, target_dt)
            
            # 确保不超出原始数据范围
            target_times = target_times[target_times <= original_times[-1]]
            target_length = len(target_times)
            
            # 计算插值索引和权重
            # 对每个目标时间点，找到最近的左侧原始时间点索引
            indices = cp.searchsorted(original_times, target_times) - 1
            indices = cp.clip(indices, 0, frames - 2)  # 确保索引有效
            
            # 计算插值权重
            alpha = (target_times - original_times[indices]) / (original_times[indices + 1] - original_times[indices])
            alpha = alpha.reshape(1, 1, -1)  # 调整形状以便广播
            
            # 将htt重塑为便于批量处理的形式
            htt_reshaped = htt_gpu.reshape(height, width, frames)
            
            # 分配结果数组
            result_gpu = cp.zeros((height, width, target_length), dtype=htt_gpu.dtype)
            
            # 批量执行线性插值
            for t in range(target_length):
                idx = indices[t]
                a = alpha[0, 0, t]
                result_gpu[:, :, t] = (1 - a) * htt_reshaped[:, :, idx] + a * htt_reshaped[:, :, idx + 1]
            
            # 将结果转回CPU - 分块处理以减少内存占用
            chunk_size = min(100, target_length)  # 每次处理100个时间帧或更少
            interpolated_htt = np.zeros((height, width, target_length), dtype=htt.dtype)

            for i in range(0, target_length, chunk_size):
                end_idx = min(i + chunk_size, target_length)
                # 分块复制，减少瞬时内存占用
                interpolated_htt[:, :, i:end_idx] = cp.asnumpy(result_gpu[:, :, i:end_idx])

            target_times_cpu = cp.asnumpy(target_times)

            # GPU垃圾回收
            del htt_gpu, original_times, target_times, indices, alpha, htt_reshaped, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            
            return interpolated_htt, target_times_cpu
            
        except (ImportError, ModuleNotFoundError):
            print("无法导入cupy或GPU不可用，回退到CPU计算...")
            use_gpu = False
    
    # CPU计算方式
    if not use_gpu:
        original_times = np.linspace(0, original_duration, frames)
        
        target_times = np.arange(0, original_duration, target_dt)
        
        target_times = target_times[target_times <= original_times[-1]]

        total_size = height * width * frames
        
        if total_size > 50_000_000:  # 约200MB内存用量的阈值，可根据实际情况调整
            interpolated_htt = np.zeros((height, width, len(target_times)), dtype=htt.dtype)
            
            from tqdm import tqdm
            for i in tqdm(range(height), desc="插值处理"):
                for j in range(width):
                    # 针对单个空间点的时间序列进行插值
                    time_series = htt[i, j, :]
                    interpolator = np.interp  # 使用numpy的内置插值
                    interpolated_htt[i, j, :] = interpolator(target_times, original_times, time_series)
        else:
            # 对于规模适中的数据，使用scipy的高效批量处理
            from scipy.interpolate import interp1d
            
            # 重塑htt以便更高效地进行插值
            reshaped_htt = htt.reshape(height * width, frames)
            
            # 创建插值函数
            interpolator = interp1d(original_times, reshaped_htt, axis=1, kind='linear', 
                                    bounds_error=False, fill_value='extrapolate')
            
            # 执行插值
            interpolated_reshaped = interpolator(target_times)
            
            # 恢复原始形状
            interpolated_htt = interpolated_reshaped.reshape(height, width, len(target_times))
        
        return interpolated_htt, target_times
    
def add_signal_to_center_points(interpolated_htt, normalised_noised_signal, cellPoint=30):
    """
    在interpolated_htt的中心区域加上normalised_noised_signal信号
    
    参数:
    interpolated_htt: 插值后的海浪高度场，形状为 [height, width, frames]
    normalised_noised_signal: 归一化后的噪声信号，形状为 [frames]
    cellPoint: 中心区域的边长（正方形区域）
    
    返回:
    numpy.ndarray: 添加信号后的海浪高度场
    """
    # 获取数组形状
    height, width, frames = interpolated_htt.shape
    
    # 确保信号长度与时间帧数匹配
    min_length = min(frames, len(normalised_noised_signal))
    if min_length < frames:
        interpolated_htt = interpolated_htt[:, :, :min_length]
    if min_length < len(normalised_noised_signal):
        normalised_noised_signal = normalised_noised_signal[:min_length]
    
    # 创建结果数组的副本，避免修改原始数据
    result_htt = interpolated_htt.copy()
    
    # 计算中心点索引
    center_i = height // 2
    center_j = width // 2
    
    # 计算中心区域的起始和结束索引
    start_i = center_i - cellPoint // 2
    end_i = start_i + cellPoint
    start_j = center_j - cellPoint // 2
    end_j = start_j + cellPoint
    
    # 确保索引在有效范围内
    start_i = max(0, start_i)
    end_i = min(height, end_i)
    start_j = max(0, start_j)
    end_j = min(width, end_j)
    
    # 在中心区域的每个点上添加信号
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            # 添加信号到每个时间点
            result_htt[i, j, :] += normalised_noised_signal
    
    # 也可以同时返回中心点的时间序列，用于后续分析
    center_signal = result_htt[center_i, center_j, :]
    center_wave = interpolated_htt[center_i, center_j, :]
    
    return result_htt, center_signal, center_wave


def save_data(htt, V, M_scale, P_scale, cover_size, second, Num, patch_size, mesh_size, deta_x):
    """
    Save the generated data to a file.
    
    Parameters:
    htt (numpy.ndarray): Height field of the ocean surface.
    V (float): Wind speed in m/s.
    M_scale (float): Scale factor for the mesh size.
    P_scale (float): Scale factor for the wave spectrum.
    cover_size (float): Cover length in meters.
    second (int): Number of seconds for simulation.
    Num (int): Number of frames per second.
    
    Returns:
    None
    """

    # Save data
    output_dir = os.path.expanduser('data/ocean_wave_data')
    os.makedirs(output_dir, exist_ok=True)

    filename = f'TimeVaryWS{int(round(V * 10)):02d}M{M_scale:03d}P{P_scale:03d}Covs{cover_size:02d}Seco{second:01d}Fps{Num:03d}.npz'
    filepath = os.path.join(output_dir, filename)

    # Save parameters
    params = {
        'windspeed': V,
        'patch_size': patch_size,
        'mesh_size': mesh_size,
        'deta_x': deta_x,
        'fps': Num
    }

    # Save data and parameters
    np.savez(filepath, htt=htt, params=params)
    print(f"Data saved to {filepath}")

    