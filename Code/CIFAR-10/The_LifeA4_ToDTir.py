import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import random

# --- 1. 宿主 (Host)：需要捕食数据来维持 L ---
class HostOrganism(nn.Module):
    def __init__(self, dna_lambda, dna_beta):
        super().__init__()
        self.dna_lambda = dna_lambda 
        self.dna_beta = dna_beta     
        self.energy = 1.0  # 初始能量
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.mu = nn.Linear(64, 256)
        self.log_var = nn.Linear(64, 256)
        self.tail = nn.Linear(256, 10)
        
    def forward(self, x):
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = self.log_var(feat).clamp(-10, 5)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return self.tail(z), mu, log_var

# --- 2. 寄生者 (Parasite)：靠宿主的错误为生 ---
class LogicParasite(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy = 0.5 
        self.generator = nn.Sequential(
            nn.Linear(10, 512), nn.GELU(),
            nn.Linear(512, 3072), nn.Tanh()
        )
    def forward(self, labels_onehot):
        return self.generator(labels_onehot).view(-1, 3, 32, 32) * 0.2

# --- 3. 生态引擎 ---
def run_starvation_ecology():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据内化
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    all_x = torch.stack([train_set[i][0] for i in range(50000)]).to(device)
    all_y = torch.tensor([train_set[i][1] for i in range(50000)]).to(device)

    # 初始种群
    hosts = [HostOrganism(1e-5, 0.1).to(device) for _ in range(10)]
    parasites = [LogicParasite().to(device) for _ in range(5)]
    
    gen = 0
    while len(hosts) > 2:
        print(f"\n[Generation {gen}] Population: Hosts={len(hosts)}, Parasites={len(parasites)}")
        
        # --- 交互演化 (捕食与反寄生) ---
        for h_idx, host in enumerate(hosts):
            # 随机遭遇寄生者
            parasite = random.choice(parasites)
            optimizer_h = torch.optim.AdamW(host.parameters(), lr=1e-3)
            optimizer_p = torch.optim.AdamW(parasite.parameters(), lr=1e-4)
            
            # 模拟一天（5个batch）的生存压力
            idx = torch.randperm(50000)[:256]
            x, y = all_x[idx], all_y[idx]
            
            # 1. 寄生者准备陷阱
            y_onehot = torch.eye(10, device=device)[y]
            noise = parasite(y_onehot)
            attacked_x = x + noise
            
            # 2. 宿主进食尝试
            logits, mu, log_var = host(attacked_x)
            ce_loss = F.cross_entropy(logits, y)
            
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                acc = (probs.argmax(1) == y).float().mean().item()
            
            # --- 能量分配逻辑 ---
            # 宿主进食：Acc越高，能量摄入越多
            feed_gain = acc * 0.8 
            # 寄生者分成：宿主错得越多，寄生者吃得越饱
            theft_loss = (1.0 - acc) * 0.2
            
            # 代谢成本：维持 L 需要消耗能量 (L 越高，扣能越多)
            entropy_vol = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            # 能量消耗公式：基础代谢 + L权重
            # 如果 L = 110，这里的 metabolic_cost 会很大
            metabolic_cost = 0.05 + (entropy_vol.item() / 1000.0)
            
            # 更新能量
            old_host_energy = host.energy
            host.energy = host.energy + feed_gain - theft_loss - metabolic_cost
            old_parasite_energy = parasite.energy
            parasite.energy = parasite.energy + theft_loss - 0.05 # 寄生者也有基础代谢
            
            # 调试信息
            print(f"Host {h_idx}: E={old_host_energy:.3f} -> {host.energy:.3f}, Acc={acc:.3f}, Feed={feed_gain:.3f}, Theft={theft_loss:.3f}, Metabolic={metabolic_cost:.3f}")
            print(f"Parasite: E={old_parasite_energy:.3f} -> {parasite.energy:.3f}")
            
            # 物理反向传播 (宿主为了生存会最小化 ce_loss 并寻找最优 entropy)
            loss_h = ce_loss - host.dna_lambda * entropy_vol
            loss_p = -ce_loss # 寄生者只想让宿主错
            
            # 梯度同步更新
            optimizer_h.zero_grad(set_to_none=True)
            optimizer_p.zero_grad(set_to_none=True)
            
            # 计算宿主梯度
            loss_h.backward(retain_graph=True)
            
            # 计算寄生者梯度
            loss_p.backward()
            
            # 更新两个模型参数
            optimizer_h.step()
            optimizer_p.step()

        # --- 3. 自然淘汰 (饥饿机制) ---
        # 能量耗尽则死亡
        surviving_hosts = [h for h in hosts if h.energy > 0]
        surviving_parasites = [p for p in parasites if p.energy > 0]
        
        # --- 4. 繁衍 (能量盈余) ---
        new_hosts = []
        for h in surviving_hosts:
            new_hosts.append(h)
            if h.energy > 1.5: # 能量充沛，分裂产生后代
                h.energy -= 0.8
                offspring = copy.deepcopy(h)
                offspring.dna_lambda *= np.random.uniform(0.7, 1.4)
                new_hosts.append(offspring)
        
        # 限制种群上限防止显存爆炸
        hosts = new_hosts[:15]
        parasites = surviving_parasites[:8]
        
        # 诊断报告
        if hosts:
            avg_L = np.mean([0.5 * torch.sum(1 + h.log_var.weight, dim=1).mean().item() for h in hosts]) # 粗略估算 L
            print(f"Top Host Energy: {hosts[0].energy:.3f} | Avg L: {avg_L:.2f} | Acc: {acc:.4f}")
        
        gen += 1
        if gen > 100: break

if __name__ == '__main__':
    run_starvation_ecology()