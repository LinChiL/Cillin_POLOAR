import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class POLOARQuantumBubble:
    def __init__(self, n_points=200, total_energy=1e6, xi=1.0, h_bar=1.0):
        """
        POLOAR宇宙模拟器
        n_points: 逻辑空间点数
        total_energy: 总能量 ∫mc² dx = E_total (守恒)
        xi: 存在耦合常数 [J·m]
        h_bar: 普朗克常数/2π [J·s] (量纲占位)
        """
        self.n = n_points
        self.xi = xi
        self.h_bar = h_bar
        self.total_energy = total_energy
        
        # 逻辑坐标 (无量纲)
        self.x = torch.arange(n_points, dtype=torch.float32)
        
        # 初始状态：寂静 (∇Ω=0 处处成立)
        # 但能量守恒不允许真正的0，所以取极小的均匀分布
        self.gradients = torch.full((n_points,), 1e-6)  # ∇Ω [m⁻¹]
        self.energies = self.xi * self.gradients  # mc² [J]
        self.energies *= (total_energy / self.energies.sum())
        self.gradients = self.energies / self.xi
        
        # 历史
        self.history_energies = []
        self.history_gradients = []
        self.history_L = []
        self.history_global_L = []
        self.history_R = []
        
    def compute_omega(self):
        """总复杂度 Ω = ∫∇Ω dx (累积)"""
        return torch.cumsum(self.gradients, dim=0)
    
    def compute_L(self):
        """杠杆率 L = log10(Ω_avg) / log10(E_avg)"""
        omega = self.compute_omega()
        omega_avg = torch.mean(omega)
        energy_avg = torch.mean(self.energies)
        L = torch.log10(omega_avg) / torch.log10(energy_avg + 1e-7)
        return L.item()
    
    def compute_global_L(self):
        """全局杠杆率：使用总Ω和总能量计算"""
        total_omega = torch.sum(self.gradients)  # 宇宙总遍历空间
        total_energy = torch.sum(self.energies)  # 宇宙总质能资产
        if total_omega > 0 and total_energy > 0:
            return torch.log10(total_omega) / torch.log10(total_energy)
        return 0.0
    
    def compute_R(self):
        """遍历速率 R = (η·mc²) / (h·∇Ω) [s⁻¹]"""
        # 简化：η=1 (全动员)
        eta = 1.0
        # 用平均能量和平均梯度估算
        energy_avg = torch.mean(self.energies)
        grad_avg = torch.mean(self.gradients)
        R = (eta * energy_avg) / (self.h_bar * grad_avg)
        return R.item()
    
    def quantum_fluctuation(self, strength=0.01):
        """
        量子涨落：∇Ω 的随机扰动
        POLOAR翻译：真空中的虚梯度对
        """
        # 随机位置
        pos = np.random.randint(0, self.n)
        # 随机方向 (±)
        sign = 1 if np.random.rand() > 0.5 else -1
        # 扰动大小：与普朗克常数相关
        delta = sign * strength * self.h_bar / self.xi
        
        # 加扰动
        self.gradients[pos] += delta
        
        # 能量守恒：重新计算能量并归一化
        self.energies = self.xi * self.gradients
        self.energies *= (self.total_energy / self.energies.sum())
        self.gradients = self.energies / self.xi
        
        return pos, delta
    
    def bubble_nucleation(self, threshold=0.01):
        """
        真真空泡成核：当某点梯度超过阈值，触发泡壁膨胀
        POLOAR翻译：局部∇Ω > 阈值 → 对称性破缺
        """
        max_grad = torch.max(self.gradients).item()
        if max_grad > threshold:
            return True
        return False
    
    def bubble_expansion(self, step, diffusion_rate=0.05):
        """
        泡壁膨胀：高斯扩散模拟
        POLOAR翻译：梯度在逻辑空间中的传播
        """
        grad_np = self.gradients.numpy()
        sigma = diffusion_rate * np.sqrt(step + 1)
        grad_smoothed = gaussian_filter1d(grad_np, sigma=sigma)
        self.gradients = torch.tensor(grad_smoothed, dtype=torch.float32)
        
        # 能量守恒
        self.energies = self.xi * self.gradients
        self.energies *= (self.total_energy / self.energies.sum())
        self.gradients = self.energies / self.xi
    
    def evolve(self, max_steps=10000, fluctuation_rate=0.1, diffusion_rate=0.05):
        """
        宇宙演化主循环——永不停歇
        """
        nucleated = False
        nucleation_step = 0
        
        for step in range(max_steps):
            # 1. 量子涨落（永远存在）
            if np.random.rand() < fluctuation_rate:
                self.quantum_fluctuation(strength=0.001)
            
            # 2. 检查是否成核（只在未成核时检查）
            if not nucleated and self.bubble_nucleation(threshold=0.01):
                nucleated = True
                nucleation_step = step
                print(f"   BUBBLE NUCLEATION at step {step}")
            
            # 3. 泡壁膨胀（如果已成核）
            if nucleated:
                # 膨胀率随步数衰减，但永不停止
                current_diffusion = diffusion_rate / (1 + 0.01 * (step - nucleation_step))
                self.bubble_expansion(step - nucleation_step, current_diffusion)
            
            # 4. 记录（每 100 步）
            if step % 100 == 0:
                L = self.compute_L()
                global_L = self.compute_global_L()
                R = self.compute_R()
                grad_std = torch.std(self.gradients).item()
                
                self.history_L.append(L)
                self.history_global_L.append(global_L)
                self.history_R.append(R)
                self.history_energies.append(self.energies.clone())
                self.history_gradients.append(self.gradients.clone())
                
                print(f"Step {step:5d} | L={L:.4f} | Global L={global_L:.4f} | R={R:.2e} | grad_std={grad_std:.2e}")
            
            # 5. 没有终止条件。永远跑下去。
        
        return self.history_L, self.history_R
    
    def plot_evolution(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # L演化
        axes[0,0].plot(self.history_L, label='Local L')
        axes[0,0].plot(self.history_global_L, label='Global L')
        axes[0,0].axhline(y=1.0, color='r', linestyle='--')
        axes[0,0].set_title("Leverage L")
        axes[0,0].legend()
        
        # R演化
        axes[0,1].plot(self.history_R)
        axes[0,1].set_yscale('log')
        axes[0,1].set_title("Traversal Rate R")
        
        # 梯度标准差演化（关键指标）
        grad_stds = [torch.std(g).item() for g in self.history_gradients]
        axes[1,0].plot(grad_stds)
        axes[1,0].set_yscale('log')
        axes[1,0].set_title("∇Ω Standard Deviation")
        axes[1,0].axhline(y=1e-3, color='r', linestyle='--', label='Threshold')
        axes[1,0].legend()
        
        # 最终梯度分布
        axes[1,1].plot(self.gradients.numpy())
        axes[1,1].set_title("Final ∇Ω Distribution")
        
        plt.tight_layout()
        plt.show()

# --- 运行POLOAR宇宙 ---
print("=" * 60)
print("POLOAR Universe Simulation")
print("=" * 60)

universe = POLOARQuantumBubble(
    n_points=200,
    total_energy=1e6,
    xi=1.0,
    h_bar=1.0
)

print("\nInitial state: SILENT (∇Ω ≈ 0)")
print("  Adding quantum fluctuations...\n")

L_hist, R_hist = universe.evolve(
    max_steps=1000,
    fluctuation_rate=0.1,
    diffusion_rate=0.05
)

print("\n" + "=" * 60)
print(f"Final L: {L_hist[-1]:.4f}")
print(f"Final Global L: {universe.history_global_L[-1]:.4f}")
print(f"Final R: {R_hist[-1]:.2e} s⁻¹")
print("=" * 60)

universe.plot_evolution()