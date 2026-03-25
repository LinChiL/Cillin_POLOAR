import torch
import numpy as np
import matplotlib.pyplot as plt

class POLOAR_ExpandingUniverse:
    def __init__(self, total_points=500, initial_energy=1000.0, xi=1.0):
        self.n = total_points
        self.xi = xi
        self.total_energy = initial_energy
        
        # 初始态：能量全部挤在中心的 10 个像素点里
        # 其他 490 个像素点是‘死’的 (nabla_Omega = 0)
        self.energies = torch.full((total_points,), 1e-9) 
        mid = total_points // 2
        self.energies[mid-5 : mid+5] = initial_energy / 10.0
        
        self.history_volume = [] # 记录有效空间大小 (∇Ω > 阈值的点数)
        self.history_max_grad = []

    def evolve(self, steps=10000):
        print("Initial Singularity ready. Releasing Logic Pressure...")
        
        for i in range(steps):
            # Law 1: PUNA 1.0
            grads = self.energies / self.xi
            
            # Law 2: 逻辑引力 (追求 Omega 最大化)
            # 我们模拟能量‘流向’逻辑潜力区的本能
            # 计算 Omega 对能量的导数，这产生了一个向外的压力
            # log(sum(grads)) 的梯度会引导能量向 0 梯度区扩散
            log_grads = torch.log(grads + 1e-9)
            # 扩散力：由于 PUNA 1.0 和 Romenda 不稳定性，能量必须向梯度低的地方逃逸
            pressure = -torch.gradient(log_grads)[0] 
            
            # 动力学更新：能量流向低梯度区
            # 这模拟了‘膨胀’：逻辑坐标被一个一个激活
            diffusion = 0.2 * (torch.roll(self.energies, 1) + torch.roll(self.energies, -1) - 2*self.energies)
            
            # 更新能量分布
            self.energies += (pressure * 2.0 + diffusion)
            
            # 约束：能量守恒
            self.energies.clamp_(min=1e-12)
            self.energies *= (self.total_energy / self.energies.sum())
            
            # 测量：什么是‘空间体积’？
            # 我们定义梯度 > 1e-4 的点为‘可观测的空间像素’
            active_volume = (grads > 1e-4).sum().item()
            
            self.history_volume.append(active_volume)
            self.history_max_grad.append(grads.max().item())
            
            if i % 200 == 0:
                print(f"Step {i:4d} | Active Volume={active_volume} | Max Grad={grads.max():.2f}")

# --- 启动演化 ---
u = POLOAR_ExpandingUniverse()
u.evolve(10000)

# 可视化：宇宙的扩张
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(u.history_volume, color='green', linewidth=2)
plt.title("Cosmic Expansion (Active Logical Pixels)")
plt.ylabel("Active Volume $V_{eff}$")
plt.xlabel("Cosmic Steps")

plt.subplot(1, 2, 2)
plt.plot(u.history_max_grad, color='red')
plt.title("Cooling (Max Gradient Decay)")
plt.ylabel("Max Gradient $\nabla\Omega_{max}$")
plt.xlabel("Cosmic Steps")

plt.tight_layout()
plt.show()