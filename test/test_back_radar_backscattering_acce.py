import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from numpy.linalg import norm
import radar
import scipy.signal as signal
from scipy.fft import fft, ifft, fftshift
import gc  # 垃圾回收
from numba import jit, prange  # 用于加速计算
import warnings

# 设置警告过滤
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 添加内存监控
def print_memory_usage():
    """打印当前内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")

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

def main():
    print("开始雷达散射仿真...")
    print_memory_usage()
    
    address = 'data/ocean_wave_data'
    name = 'TimeVaryWS10M002P008Covs10Seco5Fps050.npz'
    data_path = os.path.join(address, name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    
    htt = data['htt']  # Ocean height data over time
    para = data['params'].item()  # Parameters
    
    # 从文件名解析参数 
    V = para['windspeed']  # 风速
    Dir = 0
    patchSize = para['patch_size']
    meshSize = para['mesh_size']
    deta_x = patchSize / meshSize
    deta_x = para['deta_x']
    Fps = para['fps']
    t_num = htt.shape[2]
    t = np.arange(1, t_num + 1) / Fps
    
    # 雷达设置
    f = 77e9  # 电子波频率
    c = 3e8    # 光速
    distance_resolu = 3e8 / (2 * 4e9)
    dielectric_c = complex(9.648, -17.666)  # 介电常数
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
    patch_size_cell = beam_width * Radii
    cellPoint = round(patch_size_cell / deta_x)
    center_index = htt.shape[0] // 2
    sloc = center_index - cellPoint // 2
    
    # 模拟基准距离与随时间变化的海面高度的总距离
    base_distance = Radii # 基准距离
    
    # 预计算单位向量
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
    
    print(f"开始计算 {t_num} 个时间步...")
    print_memory_usage()
    
    # 使用批处理减少内存使用
    batch_size = min(10, t_num)  # 每次处理10个时间步
    
    for batch_start in range(0, t_num, batch_size):
        batch_end = min(batch_start + batch_size, t_num)
        print(f"处理时间步 {batch_start+1}-{batch_end}/{t_num}")
        
        for tn in range(batch_start, batch_end):
            # 提取单元格波高数据
            Hx_cell = htt[sloc-1:sloc+cellPoint-1, sloc-1:sloc+cellPoint-1, tn]
            
            # 创建位置矩阵
            Rx = np.vstack((grid_x_flat, grid_y_flat, Hx_cell.flatten()))
            
            # 使用加速函数计算距离
            radar_pos = np.array([0, 0, Radii])
            all_distance = calculate_distances_fast(radar_pos, Rx)
            
            # 并行处理距离单元
            for rn in range(range_num):
                # 距离门
                dis_low = rn * distance_resolu
                dis_high = (rn + 1) * distance_resolu
                range_here = (dis_low + dis_high) / 2
                
                # 找到当前距离门内的点
                here_loc = (all_distance > dis_low) & (all_distance < dis_high)
                
                if np.any(here_loc):  # 如果该距离门内有点
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
            print_memory_usage()
    
    print("计算完成！")
    print_memory_usage()
    
    # 可视化 - 使用更高效的绘图
    plt.style.use('default')  # 使用默认样式以减少内存使用
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 距离-时间图像
    range_cells = slice(100, 160)
    data_to_plot = 2 * 20 * np.log10(np.abs(ele_temp[:, range_cells].T))
    
    im = axes[0].imshow(data_to_plot, 
                        aspect='auto', 
                        origin='lower',
                        extent=[0, t_num, range_cells.start, range_cells.stop],
                        cmap='viridis')
    plt.colorbar(im, ax=axes[0], label='Power (dB)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Range cell')
    axes[0].set_title('3-D Echoes Simulation')
    
    # 2. 相位分析
    try:
        _, max_range_idx = radar.find_max_energy_range_bin(ele_temp.T)
        phase_range, maxloc = radar.extract_phase_from_max_range_bin(ele_temp.T, max_range_idx, range_search=10)
        unwrapped_phase, _ = radar.extract_and_unwrap_phase(ele_temp, maxloc, method='compensated', filter_noise=False, distance_resolu=distance_resolu, f=f)
        
        print(f"最大能量范围索引: {max_range_idx}, 最大位置: {maxloc}")
        np.savetxt('maxloc_data.txt', maxloc, fmt='%.6f')
        
        times = np.arange(t_num) / Fps * 1000
        
        # 检测距离单元跳变
        range_indices = []
        for ti in range(t_num):
            # 在搜索范围内找最大能量
            search_start = max(0, max_range_idx - 10)
            search_end = min(ele_temp.shape[1], max_range_idx + 10 + 1)
            
            range_slice = ele_temp[ti, search_start:search_end]
            power_slice = np.abs(range_slice)**2
            local_max_idx = np.argmax(power_slice)
            actual_range_idx = search_start + local_max_idx
            
            range_indices.append(actual_range_idx)
        
        range_indices = np.array(range_indices)
        
        # 找到距离单元跳变点
        range_jumps = np.diff(range_indices)
        jump_positions = np.where(range_jumps != 0)[0]
        
        # 检测相位大跳跃点
        phase_diffs = np.diff(unwrapped_phase)
        large_phase_jumps = np.abs(phase_diffs) > np.pi * 0.8
        phase_jump_positions = np.where(large_phase_jumps)[0]
        
        print(f"检测到 {len(jump_positions)} 个距离单元跳变")
        print(f"检测到 {np.sum(large_phase_jumps)} 个大相位跳跃")
        
        # 绘制相位分析图
        axes[1].plot(times, unwrapped_phase, 'b-', linewidth=1, label='Unwrapped Phase')
        
        # 标记距离单元跳变点（红色竖线）
        if len(jump_positions) > 0:
            for jump_pos in jump_positions:
                if jump_pos < len(times):
                    axes[1].axvline(times[jump_pos], color='red', linestyle='--', 
                                alpha=0.7, linewidth=1.5, label='Range Bin Jump' if jump_pos == jump_positions[0] else "")
        
        # 标记大相位跳跃点（橙色圆圈）
        if len(phase_jump_positions) > 0:
            jump_times = times[phase_jump_positions + 1]  # +1因为diff的索引偏移
            jump_phases = unwrapped_phase[phase_jump_positions + 1]
            axes[1].scatter(jump_times, jump_phases, color='orange', s=50, 
                        marker='o', alpha=0.8, edgecolors='black', linewidth=1,
                        label='Large Phase Jump', zorder=5)
        
        # 标记同时发生距离单元跳变和相位跳跃的点（红色圆圈）
        coincident_jumps = np.intersect1d(jump_positions, phase_jump_positions)
        if len(coincident_jumps) > 0:
            coincident_times = times[coincident_jumps + 1]
            coincident_phases = unwrapped_phase[coincident_jumps + 1]
            axes[1].scatter(coincident_times, coincident_phases, color='red', s=80, 
                        marker='o', alpha=0.9, edgecolors='darkred', linewidth=2,
                        label='Coincident Jump', zorder=6)
        
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Phase (radians)')
        axes[1].set_title(f'Phase Analysis (Range Jumps: {len(jump_positions)}, Phase Jumps: {np.sum(large_phase_jumps)})')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best', fontsize=8)
        
        # 在图上添加统计信息文本框
        info_text = f'Range bin jumps: {len(jump_positions)}\n'
        info_text += f'Phase jumps: {np.sum(large_phase_jumps)}\n'
        info_text += f'Coincident: {len(coincident_jumps)}\n'
        info_text += f'Jump rate: {len(jump_positions)/t_num*100:.1f}%'
        
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes, 
                    fontsize=8, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        # 保存跳变分析数据
        jump_analysis = {
            'range_indices': range_indices,
            'range_jump_positions': jump_positions,
            'phase_jump_positions': phase_jump_positions,
            'coincident_jumps': coincident_jumps,
            'times': times,
            'unwrapped_phase': unwrapped_phase
        }
        np.savez_compressed('phase_jump_analysis.npz', **jump_analysis)
        print(f"相位跳变分析数据已保存到 phase_jump_analysis.npz")
        
    except Exception as e:
        print(f"相位分析出错: {e}")
        import traceback
        traceback.print_exc()
        axes[1].text(0.5, 0.5, 'Phase analysis failed', 
                    transform=axes[1].transAxes, ha='center')
    
    # 3. 海面高度时间序列
    center_x = cellPoint // 2
    center_y = cellPoint // 2
    height_timeseries = htt[sloc-1+center_y, sloc-1+center_x, :]
    
    axes[2].plot(t * 1000, height_timeseries, 'b-', linewidth=1)
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Height (m)')
    axes[2].set_title('Ocean Surface Height at Center Point vs Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
    print("程序执行完毕！")
    print_memory_usage()

    
    # 保存结果
    # try:
    #     np.savez_compressed('radar_simulation_results.npz', 
    #                        ele_temp=ele_temp,
    #                        times=times if 'times' in locals() else t*1000,
    #                        height_timeseries=height_timeseries)
    #     print("结果已保存到 radar_simulation_results.npz")
    # except Exception as e:
    #     print(f"保存结果失败: {e}")
    
    input("按Enter键继续...")
    

    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        input("按Enter键退出...")