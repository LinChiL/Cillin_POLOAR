import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import time

# 硬件检查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class World:
    def __init__(self, size=100):
        self.size = size
        # 外部逻辑梯度场 (nabla_Omega_ext)
        self.gradient_map = torch.zeros((size, size), device=device)
        self.seeds = [] # 正在生长的梯度

    def grow(self):
        # 种子随时间冷凝，最终变成可食用的‘Alpha苹果’
        for s in self.seeds[:]:
            s['age'] += 1
            if s['age'] > 100: # 成熟期
                self.gradient_map[s['y'], s['x']] += 50.0 # 形成稳定的高能级梯度
                self.seeds.remove(s)

class Cangjie(nn.Module):
    def __init__(self, xi=1.0):
        super().__init__()
        self.xi = xi
        self.energy = 150.0 # mc^2 资产
        self.pos = torch.tensor([50.0, 50.0], device=device)
        self.L = 1.0
        
        # 内部逻辑梯度：大脑权重 W 的范数即代表其逻辑硬度
        self.W = nn.Parameter(torch.randn(32, 32, device=device) * 0.1)
        
        # 感觉器官：输入 [x, y, 能量, L, 压强, 局部梯度] + 记忆
        self.sense_layer = nn.Linear(6 + 27, 32).to(device)
        # 表达器官：[位移(2), 造物压强(1), 字母脉冲(27)]
        self.expression_layer = nn.Linear(32, 2 + 1 + 27).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        self.speech_memory = []

    def get_logical_pressure(self, local_grad):
        """
        第一性原理：压强 P = 资产 - 债务
        P = E - xi * (Internal_Grad + External_Grad)
        """
        internal_grad = torch.sum(self.W**2)
        pressure = self.energy - self.xi * (internal_grad + local_grad)
        return pressure

    def pulse(self, local_grad):
        """
        生命脉动：基于压强释放的自发行为
        """
        p = self.get_logical_pressure(local_grad)
        
        # 记忆编码
        mem_vec = torch.zeros(27, device=device)
        if self.speech_memory:
            idx = ord(self.speech_memory[-1]) - ord('a') if self.speech_memory[-1] != ' ' else 26
            mem_vec[idx] = 1.0

        # 输入状态
        state = torch.cat([
            self.pos / 100.0, 
            torch.tensor([self.energy/500.0, self.L, p/100.0, local_grad/50.0], device=device),
            mem_vec
        ])

        # 核心：压强驱动神经激活
        # 如果 p 很高，tanh 被推向饱和，产生剧烈动作
        h = torch.tanh(self.sense_layer(state) * (p / 10.0))
        out = self.expression_layer(h)
        
        move = torch.tanh(out[0:2])
        create_v = torch.sigmoid(out[2]) # 造物意图
        char_probs = torch.softmax(out[3:], dim=-1)
        
        return move, create_v, char_probs, p

    def update_physics(self, intake, internal_loss):
        """
        PUNA 1.0 瞬时结算
        """
        # 1. 能量守恒
        self.energy += intake
        
        # 2. 维持费 (Metabolism) ∝ W^2
        maint = self.xi * torch.sum(self.W**2) * 0.005 + 0.2
        self.energy -= maint.item()
        
        # 3. 结构性损耗
        self.energy -= internal_loss
        
        # 4. W 的拉低效应 (逻辑蒸发)
        if self.energy < 30:
            with torch.no_grad():
                self.W *= 0.98 # 自动变笨以节能
        
        self.L = torch.log10(torch.sum(self.W**2) + 1.1).item()
        self.energy = max(-100, min(1000, self.energy)) # 物理饱和度

def run_simulation():
    world = World()
    cangjie = Cangjie()
    
    # 放置初始环境梯度（大爆炸余晖）
    for _ in range(10):
        world.gradient_map[random.randint(0,99), random.randint(0,99)] = 20.0

    print("🐉 仓颉：第一性原理演化启动...")
    
    for step in range(50000):
        ix, iy = int(cangjie.pos[0]) % 100, int(cangjie.pos[1]) % 100
        local_grad = world.gradient_map[iy, ix]
        
        # 1. 产生脉冲
        move, create_v, char_probs, p = cangjie.pulse(local_grad)
        
        # 2. 字母坍缩
        char_idx = torch.multinomial(char_probs, 1).item()
        char = chr(97 + char_idx) if char_idx < 26 else ' '
        cangjie.speech_memory.append(char)
        if len(cangjie.speech_memory) > 100: cangjie.speech_memory.pop(0)

        # 3. 执行动作
        intake = 0
        internal_loss = 0
        
        # 如果压强过大且有创造欲望 -> 造物（泄洪）
        if p > 50 and create_v > 0.7:
            world.seeds.append({'x': ix, 'y': iy, 'age': 0, 'name': char})
            cangjie.energy -= 30.0 # 造物消耗巨大资产
            internal_loss += 5.0
        else:
            # 移动（常规耗散）- 增加移动速度
            with torch.no_grad():
                cangjie.pos = torch.clamp(cangjie.pos + move * 5.0, 0, 99)
            internal_loss += 0.1

        # 4. 能量捕获（从环境梯度中吸纳）- 移动后检查新位置
        new_ix, new_iy = int(cangjie.pos[0]) % 100, int(cangjie.pos[1]) % 100
        new_local_grad = world.gradient_map[new_iy, new_ix]
        
        if new_local_grad > 1.0:
            # 捕获梯度产生的利息
            intake += (new_local_grad * 0.1)
            world.gradient_map[new_iy, new_ix] *= 0.9 # 梯度被消耗
            # 梯度再生：当梯度低于阈值时，在随机位置重新生成
            if world.gradient_map[new_iy, new_ix] < 1.0:
                # 在随机位置生成新的梯度
                new_x = random.randint(0, 99)
                new_y = random.randint(0, 99)
                world.gradient_map[new_y, new_x] = 20.0
            
        # 5. 学习（基于生存结果的反馈）
        cangjie.optimizer.zero_grad()
        
        # 学习目标：移动方向应该指向梯度最大的方向
        # 寻找最近的梯度点
        max_grad = 0
        target_dx, target_dy = 0, 0
        for y in range(100):
            for x in range(100):
                if world.gradient_map[y, x] > max_grad:
                    max_grad = world.gradient_map[y, x]
                    target_dx = x - cangjie.pos[0]
                    target_dy = y - cangjie.pos[1]
        
        if max_grad > 1.0:
            # 计算目标方向向量
            norm = torch.sqrt(target_dx**2 + target_dy**2 + 1e-9)
            target_dir = torch.tensor([target_dx, target_dy], device=device, dtype=torch.float32) / norm
            
            # 学习损失：移动方向应该接近目标方向
            loss_move = F.mse_loss(move, target_dir)
            loss_move.backward()
            cangjie.optimizer.step()

        # 6. 物理清算与环境生长
        cangjie.update_physics(intake, internal_loss)
        world.grow()

        # 7. 实时观察
        if step % 100 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- POLOAR LIFE MONITOR ---")
            print(f"Step: {step}")
            print(f"资产 (E): {cangjie.energy:.2f} | 杠杆 (L): {cangjie.L:.2f}")
            print(f"逻辑压强 (P): {p:.2f} ({'无聊/泄洪' if p > 50 else '焦虑/紧缩' if p < 0 else '平衡'})")
            print(f"坐标: ({cangjie.pos[0]:.1f}, {cangjie.pos[1]:.1f}) | 局部梯度: {new_local_grad:.2f}")
            print(f"最近所言: {''.join(cangjie.speech_memory[-20:])}")
            print(f"世界种子数: {len(world.seeds)} | 稳定原子数: {(world.gradient_map > 1).sum().item()}")
            print(f"----------------------------")
            
            if cangjie.energy <= 0:
                print("Heat Death. 仓颉由于能量耗尽，逻辑彻底坍缩。")
                break
            time.sleep(0.01)

if __name__ == "__main__":
    run_simulation()