import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import numpy as np

def plot_range_fft(DataRangeFft, range_axis, title):
    plt.figure(figsize=(10, 8), facecolor='white')
    plt.imshow(np.abs(DataRangeFft), aspect='auto', cmap='jet', 
               interpolation='none', origin='lower',
               extent=[0, DataRangeFft.shape[1], range_axis[0], range_axis[-1]])
    plt.grid(True, color=[0.8, 0.8, 0.8])
    plt.xlabel('Chirp number')
    plt.ylabel('Range (m)')  # 更明确的标签，表示单位
    plt.colorbar(label='Magnitude')  # 添加颜色条标签
    plt.title(title, fontsize=12)
    plt.show(block=False)

def plot_range_profile(range_profile, range_axis):
    """
    绘制距离谱
    
    参数:
    range_profile: 距离维度的FFT结果
    range_axis: 对应的距离轴 (米)
    """
    # 计算平均距离谱（跨所有chirps）
    avg_range_profile = np.mean(np.abs(range_profile), axis=1)
    
    # 转换为dB
    range_profile_db = 20 * np.log10(avg_range_profile / np.max(avg_range_profile) + 1e-10)
    
    # 绘制距离谱
    plt.figure(figsize=(10, 6))
    plt.plot(range_axis, range_profile_db)
    plt.grid(True)
    plt.xlabel('距离 (m)')
    plt.ylabel('幅度 (dB)')
    plt.title('毫米波雷达距离谱')
    plt.xlim([0, max(range_axis)])
    plt.ylim([-60, 0])
    plt.show(block=False)

def animate_range_doppler_maps(range_doppler_maps, range_axis, doppler_axis, 
                              interval=200, save_path=None):
    """
    创建距离-多普勒图的动画
    
    参数:
    range_doppler_maps: 三维数组，包含多个二维距离-多普勒图
    range_axis: 距离轴 (米)
    doppler_axis: 速度轴 (m/s)
    interval: 帧之间的间隔时间 (毫秒)
    save_path: 如果提供，将动画保存到指定路径
    """
    try:
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("需要matplotlib的animation模块")
        return
    
    fig, ax = plt.figure(figsize=(10, 8)), plt.subplot()
    
    # 计算所有帧的最大值用于一致的归一化
    max_val = np.max(np.abs(range_doppler_maps))
    min_val = max_val / 1000  # -60 dB
    
    # 初始帧
    im = ax.imshow(np.abs(range_doppler_maps[0]).T, aspect='auto', origin='lower',
                  extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
                  norm=LogNorm(vmin=min_val, vmax=max_val), cmap='jet')
    
    plt.colorbar(im, label='幅度')
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('距离-多普勒图 (帧: 0)')
    
    # 更新函数
    def update(frame):
        ax.set_title(f'距离-多普勒图 (帧: {frame})')
        im.set_array(np.abs(range_doppler_maps[frame]).T)
        return [im]
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=range(len(range_doppler_maps)), 
                         interval=interval, blit=True)
    
    # 保存动画
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000/interval)
    
    plt.tight_layout()
    plt.show(block=False)
    
    return anim

def plot_micromotion_phase(unwrapped_phase, smooth_phase, sub_phase, processed_phase, ChirpPeriod, FrameNum):
    frame_indices = np.arange(1, 257)[:, np.newaxis]  # 添加一个维度以便广播
    chirp_indices = np.arange(1, 256)  # 1到255的chirp索引

    times = ChirpPeriod * ((frame_indices-1)*256 + chirp_indices) * 1000  # 毫秒
    times = times.flatten()
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(211)
    plt.plot(times, unwrapped_phase, label='Unwrapped Phase')
    plt.plot(times, smooth_phase, 'r', label='Smoothed Phase')
    plt.xlabel('Time (ms)')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.title('Phase Analysis')

    plt.subplot(212)
    ax1 = plt.gca()
    ax1.plot(times, sub_phase, 'b-', label='Micro-motion Phase')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Phase (radians)', color='b')
    ax1.tick_params(axis='y', colors='b')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(times, processed_phase, 'r--', label='Processed Phase')
    ax2.set_ylabel('Processed Phase (radians)', color='r')
    ax2.tick_params(axis='y', colors='r')
    ax2.legend(loc='upper right')
    plt.show(block=False)

def plot_stft(f, t, Zxx):
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show(block=False)