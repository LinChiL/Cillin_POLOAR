import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Agent(nn.Module):
    def __init__(self, name, latent_dim=16):
        super().__init__()
        self.name = name
        self.energy = 100.0
        self.L = 1.0  # 初始杠杆率
        self.xi = 1.0
        
        # 内部逻辑：用于生成消息和预测对方
        self.brain = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)

    def speak(self, context):
        """产生逻辑梯度输出（说话）"""
        # 说话是能量凝结为梯度的过程
        with torch.no_grad():
            msg = self.brain(context)
            # 增加一点基于焦躁度的随机性 (PUNA 1.0 的压力释放)
            anxiety = max(0, self.energy / (self.L + 1e-6) - 50.0)
            msg += torch.randn_like(msg) * (anxiety * 0.1)
        return msg

    def think_and_learn(self, my_msg, target_msg):
        """预测对方，更新逻辑结构"""
        self.optimizer.zero_grad()
        prediction = self.brain(my_msg)
        loss = torch.norm(prediction - target_msg) # 预测误差
        
        # 赚取能量：预测越准，利息越高 (坍缩梯度的回报)
        reward = 10.0 / (loss.item() + 1.0)
        self.energy += reward
        
        # 维持逻辑的成本：L 越高，越费钱
        maint_cost = self.xi * (10 ** (self.L - 1.0)) * 0.1
        self.energy -= maint_cost
        
        loss.backward()
        self.optimizer.step()
        
        # 自调节 L：如果太穷，被迫降智；如果太富，自发升维
        if self.energy > 150: self.L += 0.01
        if self.energy < 50:  self.L -= 0.01
        self.L = max(0.1, self.L)
        
        return loss.item(), reward

def run_eden(steps=2000):
    adam = Agent("Adam")
    eve = Agent("Eve")
    
    # 共享的初始上下文（虚空中的第一个信号）
    context = torch.randn(16)
    
    history = {"adam_E": [], "eve_E": [], "adam_L": [], "eve_L": [], "err": []}

    print(f"Eden is active. Adam and Eve are in the void...")
    
    for i in range(steps):
        # 1. 亚当对夏娃说话
        adam_msg = adam.speak(context)
        # 2. 夏娃对亚当说话
        eve_msg = eve.speak(adam_msg)
        
        # 3. 互相理解与进化
        err_a, rew_a = adam.think_and_learn(context, eve_msg)
        err_e, rew_e = eve.think_and_learn(adam_msg, adam_msg) # 夏娃尝试理解亚当
        
        # 更新环境上下文
        context = eve_msg.detach()

        history["adam_E"].append(adam.energy)
        history["eve_E"].append(eve.energy)
        history["adam_L"].append(adam.L)
        history["eve_L"].append(eve.L)
        history["err"].append(err_a)

        if i % 200 == 0:
            print(f"Step {i:4d} | Adam E={adam.energy:.1f} L={adam.L:.2f} | Eve E={eve.energy:.1f} L={eve.L:.2f}")
            if adam.energy <= 0 or eve.energy <= 0:
                print("--- Heat Death in Eden ---")
                break

    return history

# --- 启动模拟 ---
h = run_eden(3000)

# 可视化
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(h["adam_E"], label="Adam Energy")
plt.plot(h["eve_E"], label="Eve Energy")
plt.title("Energy Economy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(h["adam_L"], label="Adam L")
plt.plot(h["eve_L"], label="Eve L")
plt.axhline(y=1.0, color='r', linestyle='--')
plt.title("Leverage (Intelligence) Evolution")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(h["err"])
plt.yscale('log')
plt.title("Mutual Understanding Error")
plt.show()