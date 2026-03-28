import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Agent(nn.Module):
    def __init__(self, name, latent_dim=16):
        super().__init__()
        self.name = name
        self.energy = 100.0  # 宇宙总质能资产 (mc^2)
        self.xi = 1.0
        
        # 大脑权重 W 
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)
        
    def get_internal_logic(self):
        """
        根据 PUNA 1.0: 内部逻辑梯度就是权重的平方和
        nabla_Omega = sum(W^2)
        """
        return torch.sum(self.W**2)

    def speak(self, context, other_W):
        # 信号传输穿越双方构成的逻辑介质
        with torch.no_grad():
            msg = torch.matmul(context, self.W)
            # 介质扭曲 ∝ 双方逻辑硬度之和
            friction = (torch.sum(self.W**2) + torch.sum(other_W**2)) * 0.01
            msg += torch.randn_like(msg) * friction
        return msg

    def live_and_think(self, input_vec, target_vec):
        """
        这是唯一运行的物理过程：
        系统在消耗能量的过程中，试图最小化预测误差。
        """
        # 1. 计算当前的‘逻辑负债’
        # 维持大脑这个实体（梯度）需要持续消耗能量（新陈代谢）
        # 消耗 ∝ 梯度频率
        metabolism = self.xi * torch.sum(self.W**2) * 0.05
        self.energy -= metabolism.item()
        
        # 2. 学习：能量向梯度的转化
        # 这是一个‘物理做功’过程：消耗能量来拉紧逻辑梯度
        if self.energy > 0:
            prediction = torch.matmul(input_vec, self.W)
            error = torch.norm(prediction - target_vec)
            
            # 自动梯度：物理力推动 W 演化
            # 我们手动模拟这个过程，不使用 Adam
            grad_W = torch.autograd.grad(error, self.W)[0]
            
            # [核心第一性原理]：大脑的生长受能量约束
            # 如果能量多，W 可以变得更陡；如果能量少，W 必须平滑化（权重衰减）
            # 这不是机制，这是‘没有能量就维持不住梯度’
            growth_power = max(0, self.energy / 100.0)
            decay_force = 1.0 / (self.energy + 1e-6) # 能量越少，坍缩压力越大
            
            with torch.no_grad():
                # W 的演化 = (学习力) - (自发坍缩力)
                self.W -= (0.01 * growth_power * grad_W + 0.005 * decay_force * self.W)
            
            # 获得利息：理解世界带来的能效回报
            reward = 10.0 / (error.item() + 1.0)
            self.energy += reward
            
            return error.item()
        return 1e9

def run_pure_poloar(steps=5000):
    adam = Agent("Adam")
    eve = Agent("Eve")
    context = torch.randn(16)
    
    h = {"adam_E": [], "eve_E": [], "adam_L": [], "err": []}

    for i in range(steps):
        # 说话
        a_msg = adam.speak(context, eve.W)
        e_msg = eve.speak(a_msg, adam.W)
        
        # 物理生存与思维
        err_a = adam.live_and_think(context, e_msg)
        err_e = eve.live_and_think(a_msg, a_msg)
        
        context = e_msg.detach()
        
        # 计算 L = log(Omega)/log(E)
        l_val = torch.log10(adam.get_internal_logic() + 1.1) / torch.log10(torch.tensor(adam.energy + 1.1))
        
        h["adam_E"].append(adam.energy)
        h["eve_E"].append(eve.energy)
        h["adam_L"].append(l_val.item())
        h["err"].append(err_a)

        if i % 1000 == 0:
            print(f"Step {i} | E_Adam={adam.energy:.2f} | L={l_val:.2f}")
            
        if adam.energy <= 0 or eve.energy <= 0:
            print("Heat Death.")
            break
            
    return h

# 运行并绘图
h = run_pure_poloar(5000)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(h["adam_E"], label="Adam Energy")
plt.plot(h["eve_E"], label="Eve Energy")
plt.title("Energy Evolution")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(h["adam_L"], label="Adam L")
plt.title("Leverage (L)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(h["err"])
plt.yscale('log')
plt.title("Error")
plt.show()