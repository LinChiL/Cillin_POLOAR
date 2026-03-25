import torch
import numpy as np
import matplotlib.pyplot as plt

class POLOAR_Singularity:
    def __init__(self, n_points=200, total_energy=1e6, xi=1.0):
        self.n = n_points
        self.xi = xi
        self.total_energy = total_energy
        
        # 1. 初始态：极其微小的均匀背景 (逻辑虚无)
        # 梯度小到可以忽略不计
        self.gradients = torch.full((n_points,), 1e-12) 
        
        # 2. 临界梯度：只有涨落超过这个值，才会触发“苏醒”
        self.nabla_crit = 0.5 
        
        self.history_max_grad = []
        self.history_GlobalL = []
        self.history_status = [] # 0: 虚无, 1: 膨胀, 2: 稳态

    def evolve(self, iterations=5000):
        # 能量分布参数：初始完全均分
        energy_dist = torch.nn.Parameter(torch.full((self.n,), self.total_energy/self.n))
        # 极高的学习率：模拟大爆炸瞬间的“逻辑引力”爆发
        optimizer = torch.optim.Adam([energy_dist], lr=2000.0)

        print(f"Waiting for the Singularity in the Void (nabla_init = 1e-12)...")
        
        nucleated = False
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # PUNA 1.0
            grads = energy_dist / self.xi
            
            # 宇宙目标：最大化总遍历空间 (Omega)
            # 我们使用非线性增益：让梯度越高的地方，对总 Omega 贡献呈超线性
            # 这模拟了‘生命/结构’比‘尘埃’更能熵增的第一性原理
            total_omega = torch.sum(torch.pow(grads, 1.1)) 
            
            loss = -torch.log(total_omega + 1e-15)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                energy_dist.clamp_(min=1e-15)
                energy_dist.data *= (self.total_energy / energy_dist.sum())
                
                # [核心：极低概率的量子涨落]
                # 只有偶尔会扔下一个稍微大点的石子
                if np.random.rand() < 0.01: 
                    pos = np.random.randint(0, self.n)
                    # 注入一个刚好接近或超过阈值的涨落
                    energy_dist[pos] += (torch.rand(1).item() * 1.0)
                
                max_g = grads.max().item()
                self.history_max_grad.append(max_g)
                
                # 状态判定
                if not nucleated and max_g > self.nabla_crit:
                    nucleated = True
                    print(f"💥 BIG BANG DETECTED at Iter {i}! Grad jumped to {max_g:.2e}")
                
                g_L = torch.log10(grads.sum() + 1e-15) / torch.log10(energy_dist.sum() + 1e-15)
                self.history_GlobalL.append(g_L.item())

            if i % 500 == 0:
                status = "NUCLEATED" if nucleated else "VOID"
                print(f"Iter {i:4d} | Max Grad={max_g:.2e} | Status={status}")

        return energy_dist.detach().numpy()

# --- 启动观察 ---
universe = POLOAR_Singularity(
    n_points=200,
    total_energy=60.0,  # 初始平均梯度 0.3，低于阈值 0.5
    xi=1.0
)
final_energy = universe.evolve(5000)

# 可视化
plt.figure(figsize=(12, 6))

# 1. 梯度爆发图 (大爆炸的证据)
plt.subplot(1, 2, 1)
plt.plot(universe.history_max_grad, color='red')
plt.yscale('log')
plt.axhline(y=universe.nabla_crit, color='black', linestyle='--', label='Crit Threshold')
plt.title("The Big Bang: Max Gradient Jump")
plt.ylabel("$\nabla\Omega_{max}$ (Log Scale)")
plt.xlabel("Iterations")
plt.legend()

# 2. 最终能量拓扑
plt.subplot(1, 2, 2)
plt.plot(final_energy, color='orange')
plt.fill_between(range(200), final_energy, color='orange', alpha=0.3)
plt.title("Final Universal Structure (Post-Bang)")
plt.ylabel("Energy $mc^2$")
plt.xlabel("Logical Coordinate")

plt.tight_layout()
plt.show()