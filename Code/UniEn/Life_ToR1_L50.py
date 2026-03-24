import torch
import numpy as np
import matplotlib.pyplot as plt

class MetabolicUniverse:
    def __init__(self, n_agents=50, initial_energy=1e7):
        self.n = n_agents
        # 每个代理人的初始能量
        self.energies = torch.full((n_agents,), initial_energy)
        # 演化参数 theta
        self.thetas = torch.full((n_agents,), 0.01, requires_grad=True)
        self.history_L = []
        self.history_E = []

    def evolve(self, lr=0.01):
        # 1. 计算个体的复杂度
        # Omega = exp(theta * 8) -> 模拟更陡峭的逻辑梯度
        log_omegas = self.thetas * 8.0
        total_log_omega = torch.sum(log_omegas)
        
        # 2. 计算熵增速率 R (KPI)
        # 宇宙指令：最大化 R = (E * Harvest_Rate) / Omega
        total_energy = torch.sum(self.energies)
        
        # [核心修改：代谢增益]
        # 当 theta 增加时，系统捕获环境能量的能力呈非线性增长
        # 模拟：生命进化出了摄食/光合作用
        harvest_capability = torch.pow(self.thetas, 2.5) * 500.0
        
        log_rate = torch.log(total_energy + 1e-6) + torch.log(harvest_capability + 1e-6) - total_log_omega * 0.05
        
        loss = -torch.mean(log_rate)
        loss.backward()
        
        with torch.no_grad():
            self.thetas -= lr * self.thetas.grad
            self.thetas.clamp_(min=1e-3, max=10.0)
            self.thetas.grad.zero_()
            
            # 3. 能量动力学
            # 维护成本：随复杂度指数级增加
            maintenance = torch.exp(self.thetas * 1.2) * 20.0
            # 捕获能量：只有变复杂才能抓到环境中的能量
            captured_energy = harvest_capability * 100.0
            
            self.energies += (captured_energy - maintenance)
            self.energies.clamp_(min=0.0)
            
            # 计算全系统的杠杆率 L
            # L = log10(Total_Omega) / log10(Total_Energy)
            log10_omega_total = total_log_omega.item() / np.log(10)
            log10_budget = np.log10(torch.sum(self.energies).item() + 1e-7)
            current_L = log10_omega_total / log10_budget
            
        return current_L, torch.sum(self.energies).item()

# --- 启动上帝模式 ---
u = MetabolicUniverse(n_agents=50)
print(f"{'Step':<10} | {'L (Leverage)':<12} | {'Total Energy':<12} | {'Status'}")
print("-" * 65)

for i in range(1000):
    L, E = u.evolve()
    u.history_L.append(L)
    u.history_E.append(E)
    
    if i % 100 == 0:
        status = "LIFE (L>1)" if L > 1.0 else "MATTER"
        print(f"{i:<10} | {L:<12.4f} | {E:<12.2e} | {status}")
    
    if E <= 0:
        print("System Collapsed (Starvation).")
        break

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(u.history_L, color='crimson', linewidth=2)
plt.axhline(y=1.0, color='black', linestyle='--', label='Life Point L=1')
plt.title("Leverage L: The Metabolic Ignition")
plt.ylabel("Leverage L")
plt.xlabel("Steps")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(u.history_E, color='gold', linewidth=2)
plt.yscale('log')
plt.title("Total System Energy (Asset)")
plt.ylabel("Energy")
plt.xlabel("Steps")
plt.show()