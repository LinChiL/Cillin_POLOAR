import torch
import numpy as np
import matplotlib.pyplot as plt

class POLOAR_LivelyUniverse:
    def __init__(self, n_points=100, total_energy=500.0, xi=1.0):
        self.n = n_points
        self.xi = xi
        self.total_energy = total_energy
        
        # 1. 初始态：接近平庸但带有极小毛刺
        self.energies = torch.full((n_points,), total_energy/n_points) + torch.randn(n_points) * 0.1
        self.history_max_grad = []
        self.history_GlobalL = []

    def _apply_laws(self, noise_strength=0.5):
        """完全基于第一性原理的动态演化"""
        
        # [ Law 1: PUNA 1.0 ] 
        # 梯度由能量密度瞬时决定
        grads = self.energies / self.xi
        
        # [ Law 2: Romenda 不稳定性 / 逻辑压差驱动 ]
        # 能量不会停，它会向梯度高的地方‘坍缩’，也会因为排挤效应向外‘扩散’
        # 我们模拟这种‘寻找 Omega 最大化’的本能：
        # 计算每个点对 Omega 的贡献：d(log Omega) / dE
        # 这产生了一个自发的‘逻辑引力’
        logical_force = torch.gradient(torch.log(grads + 1e-9))[0]
        
        # 能量重排：能量随‘逻辑力’流动 + 永恒的量子背景涨落
        diffusion = 0.1 * (torch.roll(self.energies, 1) + torch.roll(self.energies, -1) - 2*self.energies)
        quantum_jitter = torch.randn(self.n) * noise_strength
        
        # 动态更新
        self.energies += (logical_force * 10.0 + diffusion + quantum_jitter)
        
        # [ 全局约束：能量守恒 ]
        self.energies.clamp_(min=1e-6)
        self.energies *= (self.total_energy / self.energies.sum())
        
        return grads

    def run(self, steps=2000):
        print(f"Universe is breathing...")
        for i in range(steps):
            grads = self._apply_laws()
            
            max_g = grads.max().item()
            g_L = torch.log10(grads.sum()) / torch.log10(torch.tensor(self.total_energy))
            
            self.history_max_grad.append(max_g)
            self.history_GlobalL.append(g_L.item())
            
            if i % 500 == 0:
                print(f"Step {i:4d} | Max Grad={max_g:.2f} | Global L={g_L:.4f}")

# --- 启动 ---
u = POLOAR_LivelyUniverse()
u.run(3000)

# 可视化：看它跳不跳
plt.figure(figsize=(12, 5))
plt.plot(u.history_max_grad, color='blue', linewidth=0.8)
plt.title("The Eternal Jitter of $\nabla\Omega_{max}$ (Lin's Second Law)")
plt.ylabel("Max Gradient")
plt.xlabel("Cosmic Steps")
plt.grid(alpha=0.3)
plt.show()