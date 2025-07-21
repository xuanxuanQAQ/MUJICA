import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


class CapillaryWaveSimulator:
    """
    使用伪谱方法的毛细波仿真器
    适合中等复杂度的仿真，能处理色散和弱非线性
    """
    
    def __init__(self, Lx=0.1, Nx=256, sigma=0.073, rho=1000, g=9.81):
        """
        参数:
        Lx: 计算域长度 (m)
        Nx: 空间网格点数
        sigma: 表面张力 (N/m)
        rho: 流体密度 (kg/m³)
        g: 重力加速度 (m/s²)
        """
        self.Lx = Lx
        self.Nx = Nx
        self.sigma = sigma 
        self.rho = rho
        self.g = g
        
        # 空间网格
        self.dx = Lx / Nx
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        
        # 频率网格
        self.k = 2 * np.pi * fftfreq(Nx, self.dx)
        self.k[0] = 1e-10  # 避免除零
        
        # 色散关系
        self.omega = np.sqrt(self.g * np.abs(self.k) + 
                           self.sigma * self.k**2 / self.rho)
    
    def initial_condition(self, wave_type='gaussian_packet'):
        """设置初始条件"""
        if wave_type == 'gaussian_packet':
            # 高斯波包
            x0 = self.Lx / 2
            sigma_x = self.Lx / 20
            k0 = 200  # 中心波数
            A = 0.002  # 振幅
            
            self.eta = A * np.exp(-((self.x - x0) / sigma_x)**2) * \
                      np.cos(k0 * (self.x - x0))
            
        elif wave_type == 'single_mode':
            # 单一模式
            k0 = 300
            A = 0.001
            self.eta = A * np.cos(k0 * self.x)
            
        elif wave_type == 'random':
            # 随机波场
            np.random.seed(42)
            phases = 2 * np.pi * np.random.random(self.Nx//2)
            amplitudes = 0.001 * np.exp(-((self.k[:self.Nx//2] - 200) / 50)**2)
            
            eta_hat = np.zeros(self.Nx, dtype=complex)
            eta_hat[1:self.Nx//2] = amplitudes[1:] * np.exp(1j * phases[1:])
            eta_hat[self.Nx//2+1:] = np.conj(eta_hat[1:self.Nx//2][::-1])
            
            self.eta = np.real(ifft(eta_hat))
    
    def linear_evolution(self, t):
        """线性演化算子"""
        eta_hat = fft(self.eta)
        eta_hat *= np.exp(-1j * self.omega * t)
        return np.real(ifft(eta_hat))
    
    def nonlinear_step(self, dt):
        """
        使用分步傅里叶方法处理弱非线性
        这里实现最简单的Zakharov方程近似
        """
        # 转到频率空间
        eta_hat = fft(self.eta)
        
        # 线性部分 (半步)
        eta_hat *= np.exp(-1j * self.omega * dt/2)
        self.eta = np.real(ifft(eta_hat))
        
        # 非线性部分 (全步) - 简化处理
        # 实际应用中这里会更复杂
        nonlinear_term = -0.5 * self.g * self.eta**2  # 简化的非线性项
        self.eta += dt * nonlinear_term
        
        # 线性部分 (半步)
        eta_hat = fft(self.eta)
        eta_hat *= np.exp(-1j * self.omega * dt/2)
        self.eta = np.real(ifft(eta_hat))
    
    def run_simulation(self, T=0.01, dt=1e-5, method='linear'):
        """
        运行仿真
        T: 总时间
        dt: 时间步长
        method: 'linear' 或 'nonlinear'
        """
        Nt = int(T / dt)
        self.time = np.linspace(0, T, Nt)
        self.eta_history = np.zeros((Nt, self.Nx))
        
        for i, t in enumerate(self.time):
            if method == 'linear':
                self.eta_history[i] = self.linear_evolution(t)
            else:
                if i == 0:
                    self.eta_history[i] = self.eta.copy()
                else:
                    self.nonlinear_step(dt)
                    self.eta_history[i] = self.eta.copy()
        
        return self.time, self.x, self.eta_history
    
    def analyze_dispersion(self):
        """分析色散关系"""
        # 理论色散关系
        k_theory = np.linspace(10, 1000, 100)
        omega_theory = np.sqrt(self.g * k_theory + 
                              self.sigma * k_theory**3 / self.rho)
        
        # 相速度和群速度
        cp_theory = omega_theory / k_theory
        cg_theory = (self.g + 3 * self.sigma * k_theory**2 / self.rho) / \
                   (2 * omega_theory)
        
        return k_theory, omega_theory, cp_theory, cg_theory
    
    def plot_results(self, save_animation=False):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 波面演化
        axes[0, 0].contourf(self.x*1000, self.time*1000, self.eta_history*1000, 
                           levels=20, cmap='RdBu_r')
        axes[0, 0].set_xlabel('位置 (mm)')
        axes[0, 0].set_ylabel('时间 (ms)')
        axes[0, 0].set_title('波面高度演化 (mm)')
        
        # 初始和最终波形
        axes[0, 1].plot(self.x*1000, self.eta_history[0]*1000, 'b-', 
                       label='初始')
        axes[0, 1].plot(self.x*1000, self.eta_history[-1]*1000, 'r-', 
                       label='最终')
        axes[0, 1].set_xlabel('位置 (mm)')
        axes[0, 1].set_ylabel('波面高度 (mm)')
        axes[0, 1].legend()
        axes[0, 1].set_title('波形对比')
        
        # 色散关系
        k_theory, omega_theory, cp_theory, cg_theory = self.analyze_dispersion()
        axes[1, 0].plot(k_theory, omega_theory, 'b-', label='频率')
        axes[1, 0].set_xlabel('波数 k (m⁻¹)')
        axes[1, 0].set_ylabel('角频率 ω (rad/s)')
        axes[1, 0].set_title('色散关系')
        axes[1, 0].grid(True)
        
        # 相速度和群速度
        axes[1, 1].plot(k_theory, cp_theory, 'b-', label='相速度')
        axes[1, 1].plot(k_theory, cg_theory, 'r-', label='群速度')
        axes[1, 1].set_xlabel('波数 k (m⁻¹)')
        axes[1, 1].set_ylabel('速度 (m/s)')
        axes[1, 1].legend()
        axes[1, 1].set_title('波速')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建仿真器
    sim = CapillaryWaveSimulator(Lx=0.05, Nx=256)
    
    # 设置初始条件
    sim.initial_condition('gaussian_packet')
    
    # 运行线性仿真
    time, x, eta_history = sim.run_simulation(T=0.008, dt=1e-5, method='linear')
    
    # 绘制结果
    sim.plot_results()
    
    print(f"仿真完成:")
    print(f"- 计算域: {sim.Lx*1000:.1f} mm")
    print(f"- 网格点数: {sim.Nx}")
    print(f"- 毛细长度: {np.sqrt(sim.sigma/(sim.rho*sim.g))*1000:.2f} mm")
    print(f"- 最小波速: {np.sqrt(2*np.sqrt(sim.g*sim.sigma/sim.rho)):.3f} m/s")