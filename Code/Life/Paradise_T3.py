import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Snake:
    def __init__(self, latent_dim=16):
        # 蛇的逻辑模式：每一秒都在微弱变化
        self.pattern = torch.randn(latent_dim)
        self.wiggle_strength = 0.5

    def wiggle(self):
        """蛇的蠕动：逻辑漂移，产生永远无法被彻底预测的梯度"""
        self.pattern += torch.randn_like(self.pattern) * self.wiggle_strength
        # 归一化防止模式飞掉
        self.pattern = self.pattern / (self.pattern.norm() + 1e-9)

class AppleTree:
    def __init__(self, latent_dim=16):
        self.secret_pattern = torch.randn(latent_dim)
        self.energy_pool = 800.0 
        self.regen_rate = 8.0

    def grow(self):
        self.energy_pool = min(1500.0, self.energy_pool + self.regen_rate)

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

    def speak(self, context):
        with torch.no_grad():
            msg = self.brain(context)
            # 焦躁度调节
            anxiety = max(0, self.energy / (self.L + 1e-6) - 50.0)
            msg += torch.randn_like(msg) * (anxiety * 0.02)
        return msg

    def think_and_learn(self, input_vec, target_vec, reward_base=10.0):
        self.optimizer.zero_grad()
        prediction = self.brain(input_vec)
        loss = torch.norm(prediction - target_vec)
        
        # 运行成本 (PUNA 1.0)
        maint_cost = self.xi * (10 ** (self.L - 1.0)) * 0.1
        self.energy -= maint_cost
        
        # 逻辑利息
        reward = reward_base / (loss.item() + 1.0)
        self.energy += reward
        
        loss.backward()
        self.optimizer.step()
        
        # L 值自演化（放缓增长，模拟对蛇的适应性）
        if self.energy > 250: self.L += 0.002 
        if self.energy < 40:  self.L -= 0.01
        self.L = max(0.5, self.L)
        
        return loss.item()

def run_eden_v3(steps=3000):
    adam = Agent("Adam")
    eve = Agent("Eve")
    tree = AppleTree()
    snake = Snake()
    
    context = torch.randn(16)
    history = {"adam_E": [], "eve_E": [], "adam_L": [], "eve_L": [], "err": []}

    print(f"Garden of Eden 3.0. The Snake is wiggling...")

    for i in range(steps):
        tree.grow()
        snake.wiggle()
        
        # 1. 采摘 (获取基础资产)
        adam.think_and_learn(torch.zeros(16), tree.secret_pattern, reward_base=5.0)
        eve.think_and_learn(torch.zeros(16), tree.secret_pattern, reward_base=5.0)
        
        # 2. 观察蛇 (获取新鲜感利息，宣泄闲置能量)
        # 只有在感到无聊（能量多）时，他们才会更频繁地观察蛇
        if adam.energy > 100: adam.think_and_learn(torch.ones(16), snake.pattern, reward_base=12.0)
        if eve.energy > 100:  eve.think_and_learn(torch.ones(16), snake.pattern, reward_base=12.0)
        
        # 3. 社会交流
        adam_msg = adam.speak(context)
        eve_msg = eve.speak(adam_msg)
        
        err = adam.think_and_learn(context, eve_msg, reward_base=15.0)
        eve.think_and_learn(adam_msg, adam_msg, reward_base=15.0)
        
        context = eve_msg.detach()

        history["adam_E"].append(adam.energy)
        history["eve_E"].append(eve.energy)
        history["adam_L"].append(adam.L)
        history["eve_L"].append(eve.L)
        history["err"].append(err)

        if i % 500 == 0:
            print(f"Step {i:4d} | Adam L={adam.L:.2f} E={adam.energy:.1f} | Eve L={eve.L:.2f} E={eve.energy:.1f}")

        if adam.energy <= 0 and eve.energy <= 0:
            print("--- Collective Heat Death ---")
            break

    return history

# 运行并绘图
h = run_eden_v3(6000)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(h["adam_E"], label="Adam")
plt.plot(h["eve_E"], label="Eve")
plt.title("Energy Economy (Snake Era)")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(h["adam_L"], label="Adam L")
plt.plot(h["eve_L"], label="Eve L")
plt.title("Leverage (Intelligence) Stability")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(h["err"])
plt.yscale('log')
plt.title("Understanding Error (Pumped)")
plt.show()