import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PhysicalSystem:
    def __init__(self, energy=1e6):
        # 初始能量 assets
        self.mc2 = energy 
        # theta 代表“开启复杂度”的程度，初始设为极低 (死物质态)
        self.theta = torch.tensor([0.01], requires_grad=True)
        self.history = {"L": [], "Rate": [], "Theta": []}

    def evolve(self, lr=0.01):
        # 使用对数空间计算，防止 10^34 这种数值爆炸
        # R = (eta * E) / (h * Omega)
        # log(R) = log(eta) + log(E) - log(h) - log(Omega)
        
        # 在 Toy 宇宙中，令 h=1, eta=theta (动员率随复杂度开启而提高)
        # 令 Omega = exp(theta * 5.0) -> 模拟状态空间随逻辑开启指数增长
        
        log_E = np.log(self.mc2)
        log_h = 0.0 # h = 1
        log_omega = self.theta * 5.0 
        
        # 宇宙 KPI：最大化熵增速率的对数
        # 我们加入一个“复杂度维持成本”：维持高 Omega 会消耗能量
        # 根据 PUNA 1.0: 能量是逻辑梯度的凝结，维持逻辑需要支出
        log_rate = log_E + torch.log(self.theta + 1e-6) - log_omega
        
        # 损失函数：负的速率（为了最大化速率）
        loss = -log_rate 
        
        # 自动微分
        loss.backward()
        
        # 手动更新 theta 并清空梯度
        with torch.no_grad():
            self.theta -= lr * self.theta.grad
            self.theta.clamp_(min=1e-5) # 不能小于零
            self.theta.grad.zero_()
            
            # 计算当前杠杆率 L
            # L = log10(Omega) / log10(Budget)
            # Budget = E / h*R. 在 Toy 宇宙简化为 L = theta_scaled / log10(E)
            current_L = (log_omega.item() / np.log(10)) / (log_E / np.log(10))
            
            # 能量消耗：复杂度越高，消耗越剧烈
            self.mc2 -= np.exp(self.theta.item()) * 10
            
        return current_L, np.exp(log_rate.item())

# --- 运行 Toy 宇宙 ---
universe_steps = 500
planet = PhysicalSystem(energy=1e8)

print(f"{'Step':<10} | {'L (Leverage)':<12} | {'Rate':<12} | {'Status'}")
print("-" * 60)

for i in range(universe_steps):
    L, rate = planet.evolve()
    planet.history["L"].append(L)
    planet.history["Rate"].append(rate)
    
    if i % 50 == 0:
        status = "LIFE (L>1)" if L > 1.0 else "MATTER"
        print(f"{i:<10} | {L:<12.4f} | {rate:<12.2e} | {status}")
        
    if planet.mc2 <= 0:
        print("Heat Death.")
        break

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(planet.history["L"], color='green', label='Leverage L')
plt.axhline(y=1.0, color='red', linestyle='--', label='Life Point L=1')
plt.title("Evolution of Leverage L")
plt.xlabel("Evolutionary Steps")
plt.ylabel("L")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(planet.history["Rate"], color='blue', label='Entropy Rate')
plt.yscale('log')
plt.title("Entropy Production Rate (KPI)")
plt.xlabel("Evolutionary Steps")
plt.ylabel("dOmega/dt")
plt.legend()
plt.tight_layout()
plt.show()