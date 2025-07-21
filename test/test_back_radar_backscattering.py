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

#  加载海洋表面数据
address = 'data/ocean_wave_data'
name = 'TimeVaryWS20M002P008Covs10Seco5Fps500.npz'
data = np.load(os.path.join(address, name), allow_pickle=True)
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
cellPoint = round(patch_size_cell  / deta_x)
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

# 计算散射
range_num = int(np.ceil((Radii + V**2/10) / distance_resolu))
ele_temp = np.zeros((t_num, range_num), dtype=complex)

for tn in range(t_num):
    # 创建网格
    xx, yy = np.meshgrid(((np.arange(1, cellPoint + 1) - 0.5 * cellPoint) * deta_x),
                         ((np.arange(1, cellPoint + 1) - 0.5 * cellPoint) * deta_x))
    
    # 提取单元格波高数据
    Hx_cell = htt[sloc-1:sloc+cellPoint-1, sloc-1:sloc+cellPoint-1, tn]
    
    # 海面平均高度 - 这会影响到距离测量
    mean_height = np.mean(Hx_cell)
    
    # 创建位置矩阵
    Rx = np.vstack((xx.flatten(), yy.flatten(), Hx_cell.flatten()))
    
    # Calculate distances from radar to all points
    radar_pos = np.array([[0], [0], [Radii]])
    all_distance = np.sqrt(np.sum((radar_pos - Rx)**2, axis=0))
    
    # Loop over range cells
    for rn in range(range_num):
        # Unit vectors for reflection
        unit_ei = unit_hi.copy()
        unit_kr = np.array([
            np.sin(theta_r) * np.cos(phi_r),
            -np.sin(theta_r) * np.sin(phi_r),
            np.cos(theta_r)
        ])  # Reflection unit vector
        
        unit_vr = np.array([
            -np.cos(theta_r) * np.cos(phi_r),
            -np.cos(theta_r) * np.sin(phi_r),
            -np.sin(theta_r)
        ])  # Reflection unit vector
        
        unit_hr = np.array([
            -np.sin(phi_r),
            np.cos(phi_r),
            0
        ])  # Reflection unit vector
        
        # Normal vector
        n = (unit_kr - unit_ki) / np.linalg.norm(unit_kr - unit_ki)
        
        # Alpha parameters
        alpha_x = (unit_kr[0] - unit_ki[0]) / (unit_kr[2] - unit_ki[2])
        alpha_y = (unit_kr[1] - unit_ki[1]) / (unit_kr[2] - unit_ki[2])
        
        # Scattering coefficient
        fr = (0.5 * np.sqrt(1 + alpha_x**2 + alpha_y**2) * 
                (-(1 - Rs) * np.dot(unit_ei, unit_hi) * np.dot(n, unit_ki) * unit_hi +
                (1 + Rs) * np.dot(unit_ei, unit_hi) * 
                np.cross(unit_kr, np.cross(n, unit_hi))))
        
        # Range gates
        dis_low = rn * distance_resolu
        dis_high = (rn + 1) * distance_resolu
        range_here = (dis_low + dis_high) / 2
        
        # Find points in current range gate
        here_loc = (all_distance > dis_low) & (all_distance < dis_high)
        rx_here = Rx[:, here_loc]
        
        if rx_here.shape[1] > 0:  # If there are points in this range
            # Fixed part of electric field
            ele_fix = (1j * kc * np.exp(1j * kc * range_here) / (2 * np.pi * range_here) *
                        (np.eye(3) - np.outer(unit_kr, unit_kr)) @ fr)
            
            # Integral part of electric field
            phase_term = 1j * kc * np.sum((unit_ki[:, np.newaxis] - unit_kr[:, np.newaxis]) * rx_here, axis=0)
            ele_inf = (np.nansum(deta_x * deta_x * np.exp(phase_term)) / 
                        (patch_size_cell * patch_size_cell))
            
            # Total electric field
            ele = ele_fix * ele_inf
            
            # Project onto receiver polarization
            ele_temp[tn, rn] = np.dot(ele, unit_hr)
            
# Visualization
plt.figure(figsize=(10, 6))
# Convert to dB and plot range-time image
range_cells = slice(100, 160)  # Equivalent to 100:160 in MATLAB
data_to_plot = 2 * 20 * np.log10(np.abs(ele_temp[:, range_cells].T))

plt.imshow(data_to_plot, 
            aspect='auto', 
            origin='lower',
            extent=[0, t_num, range_cells.start, range_cells.stop])
plt.colorbar(label='Power (dB)')
plt.xlabel('Time')
plt.ylabel('Range cell')
plt.title('3-D Echoes Simulation')
plt.tight_layout()
plt.show(block=False)

_, max_range_idx = radar.find_max_energy_range_bin(ele_temp.T)
phase_range, maxloc = radar.extract_phase_from_max_range_bin(ele_temp.T, max_range_idx, range_search=10)
unwrapped_phase, _ = radar.extract_and_unwrap_phase(phase_range)

times = np.arange(t_num) / Fps * 1000

plt.figure(figsize=(12, 10))

plt.plot(times, unwrapped_phase, label='Unwrapped Phase')
plt.xlabel('Time (ms)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.title('Phase Analysis')
plt.show(block=False)

center_x = cellPoint // 2
center_y = cellPoint // 2

height_timeseries = htt[sloc-1+center_y, sloc-1+center_x, :]

plt.figure(figsize=(12, 6))
plt.plot(t * 1000, height_timeseries, 'b-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Height (m)')
plt.title('Ocean Surface Height at Center Point vs Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

input("Press Enter to continue...")