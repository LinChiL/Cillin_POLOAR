import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Agent(nn.Module):
    def __init__(self, name, latent_dim=16):
        super().__init__()
        self.name = name
        self.energy = 100.0
        self.L = 1.0 
        self.xi = 1.0
        self.brain = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)

    def speak(self, context, other_L):
        """带有介质扭曲的语言输出"""
        with torch.no_grad():
            msg = self.brain(context)
            
            # 信号精细度 ∝ 自己的 L
            # 介质扭曲 ∝ 双方的 L
            # 这里的关键是：L 越高，信号越像‘精密仪器’，稍微碰一下就坏了
            signal_fragility = self.L
            noise_level = (self.L + other_L) * 0.1
            
            # 最终噪声：L 越高，不仅水更浑，信号也更怕水
            noise = torch.randn_like(msg) * (noise_level * signal_fragility)
            msg += noise
            
            return msg

    def think_and_learn(self, input_vec, target_vec, reward_base=15.0):
        self.optimizer.zero_grad()
        prediction = self.brain(input_vec)
        raw_error = torch.norm(prediction - target_vec)
        
        # [核心：高智商的解码红利 vs 脆性成本]
        # 你的智力 L 是你的放大镜。它能看清信号，但也会看清噪声。
        # 最终理解误差 = 原始误差 / (自己的 L) + (噪声引起的幻觉)
        # 这里模拟：L 越高，你对‘对’的定义越严苛
        effective_error = raw_error * (self.L ** 1.5) # L 越高，误差的‘心理权重’越大
        
        # 维持逻辑的指数级成本
        maint_cost = self.xi * (10 ** (self.L - 1.0)) * 0.1
        self.energy -= maint_cost
        
        # 逻辑利息：由于有介质扭曲，Error 很难归零，保证了永远有‘钱’赚
        reward = reward_base / (effective_error.item() + 1.0)
        reward = min(reward, 50.0)  # 限制奖励大小，避免能量爆炸
        self.energy += reward
        self.energy = min(self.energy, 1000.0)  # 设置能量上限，避免数值溢出
        
        effective_error.backward()
        self.optimizer.step()
        
        # L 的自调节（有了介质摩擦，L 应该会更稳定）
        if self.energy > 150: self.L += 0.001  # 降低L增长速率
        if self.energy < 40:  self.L -= 0.02
        self.L = max(0.8, self.L)
        
        return raw_error.item()

def run_eden_v4(steps=2000):
    adam = Agent("Adam")
    eve = Agent("Eve")
    
    context = torch.randn(16)
    history = {"adam_E": [], "eve_E": [], "adam_L": [], "eve_L": [], "err": []}

    print(f"Garden of Eden 4.0. The Medium is the Message...")

    for i in range(steps):
        # 1. 亚当和夏娃跨越介质说话
        adam_msg = adam.speak(context, eve.L)
        eve_msg = eve.speak(adam_msg, adam.L)
        
        # 2. 学习（对抗扭曲）
        err = adam.think_and_learn(context, eve_msg)
        eve.think_and_learn(adam_msg, adam_msg)
        
        context = eve_msg.detach()

        history["adam_E"].append(adam.energy)
        history["eve_E"].append(eve.energy)
        history["adam_L"].append(adam.L)
        history["eve_L"].append(eve.L)
        history["err"].append(err)

        if i % 500 == 0:
            print(f"Step {i:4d} | Adam L={adam.L:.2f} E={adam.energy:.1f} | Eve L={eve.L:.2f} E={eve.energy:.1f}")

        if adam.energy <= 0 or eve.energy <= 0:
            print("--- Final Settlement ---")
            break

    return history

# 运行并绘图
h = run_eden_v4(5000)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(h["adam_E"], label="Adam")
plt.plot(h["eve_E"], label="Eve")
plt.title("Energy Economy (Friction Era)")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(h["adam_L"], label="Adam L")
plt.plot(h["eve_L"], label="Eve L")
plt.title("Leverage (Intelligence) Balance")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(h["err"])
plt.yscale('log')
plt.title("The Eternal Misunderstanding (Noise)")
plt.show()