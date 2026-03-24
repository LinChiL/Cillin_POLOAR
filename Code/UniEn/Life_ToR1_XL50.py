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
    
    def evolve(self, max_steps=1000, fluctuation_rate=0.1, diffusion_rate=0.05):
        """
        宇宙演化主循环
        """
        nucleated = False
        nucleation_step = 0
        
        for step in range(max_steps):
            # 1. 量子涨落（一直存在）
            if np.random.rand() < fluctuation_rate:
                pos, delta = self.quantum_fluctuation(strength=0.001)
            
            # 2. 检查是否成核
            if not nucleated and self.bubble_nucleation(threshold=0.01):
                nucleated = True
                nucleation_step = step
                print(f"   BUBBLE NUCLEATION at step {step}")
                print(f"   ∇Ω_max = {torch.max(self.gradients).item():.4f}")
            
            # 3. 泡壁膨胀（如果已成核）
            if nucleated:
                self.bubble_expansion(step - nucleation_step, diffusion_rate)
            
            # 4. 记录
            if step % 50 == 0:
                L = self.compute_L()
                global_L = self.compute_global_L()
                R = self.compute_R()
                self.history_L.append(L)
                self.history_global_L.append(global_L)
                self.history_R.append(R)
                self.history_energies.append(self.energies.clone())
                self.history_gradients.append(self.gradients.clone())
                
                status = "NUCLEATED" if nucleated else "SILENT"
                print(f"Step {step:4d} | L={L:.4f} | Global L={global_L:.4f} | R={R:.2e} | {status}")
            
            # 5. 终止条件：泡充满宇宙（梯度均匀化）
            if nucleated and step - nucleation_step > 200:
                grad_std = torch.std(self.gradients).item()
                if grad_std < 1e-3:
                    print(f"   BUBBLE FILLED UNIVERSE at step {step}")
                    break
        
        return self.history_L, self.history_R
    
    def plot(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # L演化
        axes[0,0].plot(self.history_L, color='purple', label='Local L')
        axes[0,0].plot(self.history_global_L, color='red', label='Global L')
        axes[0,0].axhline(y=1.0, color='green', linestyle='--', label='Life Point L=1')
        axes[0,0].set_title("Leverage L Evolution")
        axes[0,0].set_ylabel("L")
        axes[0,0].legend()
        
        # R演化
        axes[0,1].plot(self.history_R, color='blue')
        axes[0,1].set_title("Traversal Rate R Evolution")
        axes[0,1].set_ylabel("R [s⁻¹]")
        axes[0,1].set_yscale('log')
        
        # 能量分布快照
        snapshots = [0, len(self.history_energies)//3, 2*len(self.history_energies)//3, -1]
        for i, snap in enumerate(snapshots):
            if i >= 3:
                continue
            ax = axes[0,2] if i==0 else axes[1,0] if i==1 else axes[1,1]
            e = self.history_energies[snap].numpy()
            ax.bar(range(len(e)), e, color='orange', alpha=0.7, width=1)
            ax.set_title(f"Energy Distribution (Step {snap*50})")
            ax.set_ylabel("mc² [J]")
        
        # 最终梯度分布
        ax = axes[1,2]
        grad = self.gradients.numpy()
        ax.plot(grad, color='green')
        ax.set_title("Final ∇Ω Distribution")
        ax.set_ylabel("∇Ω [m⁻¹]")
        ax.set_xlabel("Logical Coordinate")
        
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

universe.plot()