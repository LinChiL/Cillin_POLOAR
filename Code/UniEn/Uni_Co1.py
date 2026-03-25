import torch
import numpy as np
import matplotlib.pyplot as plt

class POLOAR_Origin:
    def __init__(self, n_points=100, total_energy=1e5, xi=1.0):
        self.n = n_points
        self.xi = xi
        self.total_energy = total_energy
        
        # 1. 初始态：逻辑近乎寂静 (∇Ω 极小且均匀)
        self.gradients = torch.full((n_points,), 1e-6) 
        self.history_L = []
        self.history_GlobalL = []
        self.history_std = []

    def _sync_puna(self):
        """核心约束：PUNA 1.0 (mc² = ξ · ∇Ω) + 能量守恒"""
        # 能量分布随梯度凝结
        energies = self.xi * self.gradients
        # 强制能量守恒：宇宙总资产是锁死的
        scale = self.total_energy / (energies.sum() + 1e-12)
        energies *= scale
        # 梯度随能量重新对齐 (反向修正梯度)
        self.gradients = energies / self.xi
        return energies

    def evolve(self, iterations=2000, fluctuation_prob=0.05):
        # 使用 Adam 模拟宇宙‘自发’寻找熵增方向的倾向
        # 这里优化的是能量分布，目标是最大化总 Ω
        energy_dist = torch.nn.Parameter(torch.full((self.n,), self.total_energy/self.n))
        optimizer = torch.optim.Adam([energy_dist], lr=500.0)

        print(f"Observing the Void...")
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # PUNA 1.0 映射：梯度由能量密度决定
            grads = energy_dist / self.xi
            # 累积复杂度 Ω = ∫∇Ω (在环形逻辑空间，取该点对整体的贡献)
            # 我们用 log(sum(grads)) 代表总遍历体积
            total_omega = torch.sum(grads)
            loss = -torch.log(total_omega + 1e-9) 
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # 约束：能量守恒
                energy_dist.clamp_(min=1e-8)
                energy_dist.data *= (self.total_energy / energy_dist.sum())
                
                # [自发大爆炸机制]：量子涨落
                # 在 ∇Ω=0 的背景里随机投下一个虚梯度，看它能否引发连锁坍缩
                if np.random.rand() < fluctuation_prob:
                    pos = np.random.randint(0, self.n)
                    energy_dist[pos] += (torch.randn(1).item() * 50.0)
                
                # 计算物理量
                # Global L = log(sum(grad)) / log(sum(energy))
                g_L = torch.log10(grads.sum() + 1e-9) / torch.log10(energy_dist.sum() + 1e-9)
                # Local L: 各点的杠杆率均值
                l_L = torch.mean(torch.log10(grads + 1e-9) / torch.log10(energy_dist.mean() + 1e-9))
                
                self.history_GlobalL.append(g_L.item())
                self.history_L.append(l_L.item())
                self.history_std.append(grads.std().item())

            if i % 200 == 0:
                print(f"Iter {i:4d} | Global L={g_L:.4f} | Avg Local L={l_L:.4f} | max_grad={grads.max():.2e} | std={grads.std():.2e}")

        return energy_dist.detach().numpy()

# --- 启动观察 ---
universe = POLOAR_Origin(n_points=100, total_energy=1e5)
final_energy = universe.evolve(3000)

# 可视化
plt.figure(figsize=(15, 5))

# 1. L 值的演化 (看它是否回不去 0)
plt.subplot(1, 3, 1)
plt.plot(universe.history_L, color='purple', alpha=0.7, label="Local L")
plt.plot(universe.history_GlobalL, color='red', linewidth=2, label="Global L")
plt.axhline(y=1.0, color='green', linestyle='--')
plt.title("Leverage Evolution (Genesis)")
plt.legend()

# 2. 逻辑不均匀性 (林氏振幅)
plt.subplot(1, 3, 2)
plt.plot(universe.history_std, color='blue')
plt.yscale('log')
plt.title("The Unstoppable Jitter (Entropy Decay)")
plt.ylabel("Grad Std Deviation")

# 3. 最终能量分布
plt.subplot(1, 3, 3)
plt.bar(range(100), final_energy, color='orange')
plt.title("Final Energy distribution (PUNA 1.0)")
plt.show()