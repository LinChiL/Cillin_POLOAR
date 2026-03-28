import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class POLOAR_Brain(nn.Module):
    def __init__(self, input_dim=4, latent_dim=12, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.ln = nn.LayerNorm(32)
        self.mu = nn.Linear(32, latent_dim)
        self.log_var = nn.Linear(32, latent_dim)
        self.out = nn.Linear(latent_dim, output_dim)

    def forward(self, x, lambda_ent=0.1):
        h = F.gelu(self.ln(self.fc1(x)))
        mu = self.mu(h)
        log_var = torch.clamp(self.log_var(h), -5, 1)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std * lambda_ent
        entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi))
        return torch.tanh(self.out(z)), entropy

class Agent:
    def __init__(self, name):
        self.name = name
        self.reset_physical() # 初始化物理状态
        self.brain = POLOAR_Brain(input_dim=4)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.005)

    def reset_physical(self):
        """死亡后的‘转世’：重置肉身，保留灵魂(brain)"""
        self.pos = torch.rand(2) * 100.0
        self.energy = 500.0
        self.is_alive = True

    def process(self, obs, signal):
        input_vec = torch.cat([obs, signal])
        out, ent = self.brain(input_vec)
        return out[:2], out[2:], ent

def run_evolutionary_eden(total_generations=50):
    adam = Agent("Adam")
    eve = Agent("Eve")
    
    print(f"🌍 POLOAR Evolutionary Eden Started.")
    print(f"Memory integration active. Natural selection by energy liquidation.")

    for gen in range(total_generations):
        apple_pos = torch.rand(2) * 100.0
        adam.reset_physical()
        eve.reset_physical()
        
        step = 0
        apples_eaten = 0
        
        # 当前代际的生命周期
        while adam.is_alive and eve.is_alive and step < 5000:
            adam.optimizer.zero_grad()
            eve.optimizer.zero_grad()

            # 1. 逻辑交互
            obs_a = (apple_pos - adam.pos).detach() / 50.0
            a_voice, _, a_ent = adam.process(obs_a, torch.zeros(2))
            _, e_move, e_ent = eve.process(torch.zeros(2), a_voice)
            
            # 2. 物理执行
            move_vec = e_move * 4.0
            with torch.no_grad():
                eve.pos = torch.clamp(eve.pos + move_vec.detach(), 0, 100)

            # 3. 损失计算 (生存压强)
            real_dist = torch.norm(eve.pos - apple_pos)
            survival_pressure = 1000.0 / (min(adam.energy, eve.energy) + 1e-6)
            
            # 惩罚：距离 + 压强 - 犹豫奖励
            loss = real_dist + survival_pressure - 0.1 * (a_ent + e_ent)
            
            loss.backward()
            adam.optimizer.step()
            eve.optimizer.step()

            # 4. 能量清算 (PUNA 1.0)
            maint = 0.8 + (a_ent.item() + e_ent.item()) * 0.02
            adam.energy -= maint
            eve.energy -= maint

            # 吃到苹果
            if real_dist.item() < 8.0:
                apples_eaten += 1
                reward = 200.0
                adam.energy += reward * 0.4
                eve.energy += reward * 0.6
                apple_pos = torch.rand(2) * 100.0

            # 判定死亡
            if adam.energy <= 0 or eve.energy <= 0:
                adam.is_alive = False
                eve.is_alive = False
            
            step += 1

        print(f"GEN {gen:2d} | Steps: {step:4d} | Apples: {apples_eaten:2d} | Final Press: {survival_pressure:.1f}")
        
        # 进化策略：如果这一代太笨（一个苹果都没吃到），进行微小的‘灵魂突变’
        if apples_eaten == 0:
            with torch.no_grad():
                for param in adam.brain.parameters():
                    param.add_(torch.randn_like(param) * 0.02)
                for param in eve.brain.parameters():
                    param.add_(torch.randn_like(param) * 0.02)

if __name__ == "__main__":
    run_evolutionary_eden(100)