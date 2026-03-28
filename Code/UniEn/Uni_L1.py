import torch
import numpy as np
import matplotlib.pyplot as plt

class POLOAR_ParticlePhysics:
    def __init__(self, n_points=200, total_energy=2000.0, xi=1.0):
        self.n = n_points
        self.xi = xi
        self.total_energy = total_energy
        
        # 初始：绝对寂静
        self.energies = torch.full((n_points,), total_energy/n_points)
        
        # 粒子阈值 (普朗克门槛)
        self.nabla_crit = 50.0 
        
        # 追踪器
        self.lifetimes = torch.zeros(n_points) # 记录每个格点处于高能态的时间
        self.particle_count = 0
        self.history_particles = []

    def step(self, dt=0.1):
        # 1. PUNA 1.0
        grads = self.energies / self.xi
        
        # 2. 逻辑动力学 (关键：非线性凝聚)
        # 能量倾向于向梯度高的地方聚集（引力），同时向外扩散（压力）
        # 这里的 grads**2 是非线性凝聚项，是产生孤子的核心
        logical_attraction = torch.gradient(grads**2)[0] 
        diffusion = 0.5 * (torch.roll(self.energies, 1) + torch.roll(self.energies, -1) - 2*self.energies)
        
        # 3. 永恒的量子涨落 (虚梯度对)
        fluctuation = torch.randn(self.n) * 2.0
        
        # 更新分布
        self.energies += (logical_attraction * 0.01 + diffusion + fluctuation)
        self.energies.clamp_(min=1e-6)
        self.energies *= (self.total_energy / self.energies.sum())
        
        # 4. 粒子捕获逻辑
        # 如果某点梯度连续多帧超过 nabla_crit，判定为“实粒子”
        active_mask = (grads > self.nabla_crit)
        self.lifetimes[active_mask] += 1
        self.lifetimes[~active_mask] = 0 # 虚粒子一旦跌落就清零
        
        # 判定实粒子：连续存在超过 50 帧
        real_particles = (self.lifetimes > 50).sum().item()
        return grads, real_particles

    def run_experiment(self, steps=2000):
        print(f"Collider active. Searching for real particles (Threshold={self.nabla_crit})...")
        for i in range(steps):
            grads, count = self.step()
            self.history_particles.append(count)
            
            if i % 500 == 0:
                print(f"Step {i:4d} | Detected Real Particles: {count}")
                
        return grads

# --- 启动捕获 ---
collider = POLOAR_ParticlePhysics()
final_grads = collider.run_experiment(3000)

# 可视化
plt.figure(figsize=(15, 5))

# 1. 粒子产生历程
plt.subplot(1, 3, 1)
plt.plot(collider.history_particles, color='green')
plt.title("Real Particle Count (Stability Search)")
plt.ylabel("Count")
plt.xlabel("Time Steps")

# 2. 最终梯度剖面图 (寻找孤子)
plt.subplot(1, 3, 2)
plt.plot(final_grads.numpy(), color='blue')
plt.axhline(y=collider.nabla_crit, color='red', linestyle='--', label='Mass Threshold')
plt.title("Final Gradient Profile")
plt.legend()

# 3. 粒子相图
plt.subplot(1, 3, 3)
# 找出所有‘卡住’的粒子位置
particles_pos = torch.where(collider.lifetimes > 50)[0]
if len(particles_pos) > 0:
    for pos in particles_pos:
        # 分析该位置的形状
        shape = final_grads[max(0, pos-5):min(200, pos+5)]
        plt.plot(shape.numpy(), label=f"P_{pos}")
plt.title("Captured Particle Micro-Structures")
plt.show()