import numpy as np
import scipy.io as sio
import os
from numpy.linalg import norm
from numba import jit
import gc 

def simulate_radar_data(num_samples, num_chirps, targets):
    """
    模拟生成毫米波雷达数据
    
    参数:
    num_samples: 每个chirp的采样点数
    num_chirps: chirp的数量
    targets: 目标列表，每个目标是一个元组 (距离，RCS)
    
    返回:
    adc_data: 模拟的ADC数据
    """
    # 创建空的ADC数据数组
    adc_data = np.zeros((num_chirps, num_samples), dtype=complex)
    
    # 雷达参数
    c = 3e8  # 光速 (m/s)
    bandwidth = 4e9  # 4 GHz带宽
    chirp_time = 40e-6  # 40 us chirp时间
    sample_rate = num_samples / chirp_time
    wavelength = c / 77e9  # 77 GHz对应的波长
    
    # 生成时间轴
    t = np.linspace(0, chirp_time, num_samples)
    
    # 为每个目标添加回波
    for target_range, target_rcs in targets:
        # 计算目标的时间延迟
        delay = 2 * target_range / c
        
        # 计算目标的中频信号
        if_freq = 2 * bandwidth * target_range / (c * chirp_time)
        
        # 计算目标回波的相位
        phase = 2 * np.pi * if_freq * t
        
        # 添加目标回波到ADC数据（考虑幅度衰减）
        amplitude = np.sqrt(target_rcs) / (target_range**2)
        for i in range(num_chirps):
            # 添加一些随机相位变化以模拟不同chirp
            random_phase = np.random.uniform(0, 2*np.pi)
            adc_data[i, :] += amplitude * np.exp(1j * (phase + random_phase))
    
    # 添加噪声
    noise_power = 0.01
    noise = np.sqrt(noise_power/2) * (np.random.randn(*adc_data.shape) + 1j * np.random.randn(*adc_data.shape))
    adc_data += noise
    
    return adc_data

@jit(nopython=True)
def calculate_distances_fast(radar_pos, points):
    """使用numba加速距离计算"""
    n_points = points.shape[1]
    distances = np.zeros(n_points)
    for i in range(n_points):
        dx = radar_pos[0] - points[0, i]
        dy = radar_pos[1] - points[1, i]
        dz = radar_pos[2] - points[2, i]
        distances[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    return distances

@jit(nopython=True)
def calculate_phase_terms(k_diff, points):
    """使用numba加速相位项计算"""
    n_points = points.shape[1]
    phase_terms = np.zeros(n_points, dtype=np.complex128)
    for i in range(n_points):
        phase = (k_diff[0] * points[0, i] + 
                k_diff[1] * points[1, i] + 
                k_diff[2] * points[2, i])
        phase_terms[i] = np.exp(1j * phase)
    return phase_terms

def ocean_surface_radar_backscattering(htt, V, Dir, pacht_size, mesh_size, deta_x, Fps):
    """
    模拟海洋表面对毫米波雷达信号的电磁波散射过程。
    
    该函数基于物理光学原理和海洋表面高度数据，计算雷达信号在海面上的散射特性。
    通过分析入射波与反射波的相互作用，模拟获取随时间变化的海面散射电场强度。
    
    参数:
        address (str): 海洋表面高度数据文件路径
        name (str): 海洋表面高度数据文件名
        
    返回:
        tuple: 包含以下元素的元组:
            - ele_receive (ndarray): 接收到的电场强度随时间的变化
            - time_series (ndarray): 对应的时间序列
    """
    t_num = htt.shape[2]
    time_series = np.arange(1, t_num + 1) / Fps

    #  雷达设置
    f = 77e9  # 电子波频率
    c = 3e8    # 光速
    distance_resolu = 3e8 / (2 * 4e9)
    dielectric_c  = complex(9.648, -17.666)  # 介电常数
    kc = 2 * np.pi * f / 3e8
    
    # 入射角和反射角设置
    thetaI = 0 / 180 * np.pi  # 入射角 -90~ 0 ~90
    phiI = 0 / 180 * np.pi  # 方向; 入射方位角
    theta_r = 0 / 180 * np.pi  # 反射角
    phi_r = 0 / 180 * np.pi  # 方向; 反射方位角

    # 计算s波的反射系数
    Rs = (np.cos(thetaI) - np.sqrt(dielectric_c - np.sin(thetaI)**2)) / \
        (np.cos(thetaI) + np.sqrt(dielectric_c - np.sin(thetaI)**2))

    Radii = 5  # 反射雷达半径（米）
    beam_width = 10 / 180 * np.pi  # 波束宽度
    patch_size_cell  = beam_width * Radii
    cellPoint = round(patch_size_cell / deta_x)
    center_index = htt.shape[0] // 2
    sloc = center_index - cellPoint // 2
    
    # 模拟基准距离与随时间变化的海面高度的总距离
    base_distance = Radii # 基准距离

    # 入射单位向量
    unit_ki = np.array([np.sin(thetaI) * np.cos(phiI), 
                        -np.sin(thetaI) * np.sin(phiI), 
                        -np.cos(thetaI)])
    unit_vi = np.array([-np.cos(thetaI) * np.cos(phiI), 
                        -np.cos(thetaI) * np.sin(phiI), 
                        -np.sin(thetaI)])
    unit_hi = np.array([-np.sin(phiI), np.cos(phiI), 0])
    
    unit_kr = np.array([
        np.sin(theta_r) * np.cos(phi_r),
        -np.sin(theta_r) * np.sin(phi_r),
        np.cos(theta_r)
    ])
    
    unit_hr = np.array([
        -np.sin(phi_r),
        np.cos(phi_r),
        0
    ])
    
    # 预计算固定参数
    n = (unit_kr - unit_ki) / np.linalg.norm(unit_kr - unit_ki)
    alpha_x = (unit_kr[0] - unit_ki[0]) / (unit_kr[2] - unit_ki[2])
    alpha_y = (unit_kr[1] - unit_ki[1]) / (unit_kr[2] - unit_ki[2])
    k_diff = kc * (unit_ki - unit_kr)
    
    # 预计算散射系数
    fr = (0.5 * np.sqrt(1 + alpha_x**2 + alpha_y**2) * 
            (-(1 - Rs) * np.dot(unit_hi, unit_hi) * np.dot(n, unit_ki) * unit_hi +
            (1 + Rs) * np.dot(unit_hi, unit_hi) * 
            np.cross(unit_kr, np.cross(n, unit_hi))))
    
    # 预计算网格
    grid_x, grid_y = np.meshgrid(
        ((np.arange(1, cellPoint + 1) - 0.5 * cellPoint) * deta_x),
        ((np.arange(1, cellPoint + 1) - 0.5 * cellPoint) * deta_x)
    )
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    # 计算散射
    range_num = int(np.ceil((Radii + V**2/10) / distance_resolu))
    ele_temp = np.zeros((t_num, range_num), dtype=complex)
    
    # 使用批处理减少内存使用
    batch_size = min(10, t_num)

    for batch_start in range(0, t_num, batch_size):
        batch_end = min(batch_start + batch_size, t_num)
        
        for tn in range(batch_start, batch_end):
            
            # 提取单元格波高数据
            Hx_cell = htt[sloc-1:sloc+cellPoint-1, sloc-1:sloc+cellPoint-1, tn]
            
            # 创建位置矩阵
            Rx = np.vstack((grid_x_flat, grid_y_flat, Hx_cell.flatten()))
            
            # 使用加速函数计算距离
            radar_pos = np.array([0, 0, Radii])
            all_distance = calculate_distances_fast(radar_pos, Rx)
            
            # Loop over range cells
            for rn in range(range_num):                
                # 距离门
                dis_low = rn * distance_resolu
                dis_high = (rn + 1) * distance_resolu
                range_here = (dis_low + dis_high) / 2
                
                # Find points in current range gate
                here_loc = (all_distance > dis_low) & (all_distance < dis_high)
                
                if np.any(here_loc): # If there are points in this range
                    rx_here = Rx[:, here_loc]
                    
                    # 固定部分的电场
                    ele_fix = (1j * kc * np.exp(1j * kc * range_here) / (2 * np.pi * range_here) *
                                (np.eye(3) - np.outer(unit_kr, unit_kr)) @ fr)
                    
                    # 使用加速函数计算相位项
                    phase_terms = calculate_phase_terms(k_diff, rx_here)
                    ele_inf = (np.sum(phase_terms) * deta_x * deta_x / 
                                (patch_size_cell * patch_size_cell))
                    
                    # 总电场
                    ele = ele_fix * ele_inf
                    
                    # 投影到接收极化
                    ele_temp[tn, rn] = np.dot(ele, unit_hr)
        
        # 定期清理内存
        if batch_start % (batch_size * 5) == 0:
            gc.collect()
    
    return ele_temp, time_series, distance_resolu, f


# 示例：如何使用上述函数
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import radar
    import plot
    import signal

    # 雷达参数
    num_samples = 512  # 每个chirp的采样点数
    num_chirps = 128   # chirp的数量
    sample_rate = 12.8e6  # 12.8 MHz采样率
    bandwidth = 4e9    # 4 GHz带宽
    center_freq = 77e9  # 77 GHz中心频率
    
    # 设置模拟目标：(距离，雷达散射截面积)
    targets = [
        (5.0, 1.0),   # 5米处的目标
        (10.0, 2.0),  # 10米处的目标
        (15.0, 0.5)   # 15米处的目标
    ]
    
    # 模拟雷达数据
    adc_data = simulate_radar_data(num_samples, num_chirps, targets)
    
    # 执行Range-FFT
    range_profile, range_axis = radar.range_fft(adc_data, num_samples, num_chirps, sample_rate, bandwidth, center_freq)
    
    # 绘制距离谱
    plot.plot_range_profile(range_profile, range_axis)
    
    # 打印检测到的峰值
    avg_profile = np.mean(np.abs(range_profile), axis=0)
    peaks, _ = signal.find_peaks(avg_profile, height=np.max(avg_profile)/10)
    
    print("检测到的目标:")
    for peak in peaks:
        print(f"距离: {range_axis[peak]:.2f} 米, 幅度: {20*np.log10(avg_profile[peak]):.2f} dB")