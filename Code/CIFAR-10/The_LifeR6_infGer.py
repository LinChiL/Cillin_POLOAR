import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import os

# [物理增强] 开启 CUDNN 极致动员
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- 掠食者肉身：APEX V3 强化型 ---
class CifarPredatorV3(nn.Module):
    def __init__(self, lambda_dna, latent_dim=256):
        super().__init__()
        self.dna = lambda_dna  # 核心基因
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x, noise_std=0.1):
        if self.training:
            x = x + noise_std * torch.randn_like(x)
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = torch.clamp(self.log_var(feat), -10, 5)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return self.tail(z), mu, log_var

# --- 进化算子 ---
def get_fitness(acc, rob):
    return (acc ** 2) * rob # 掠食者权重：极其看重逻辑的准确性，同时要求鲁棒性

def poloar_l(mu, log_var, loss, energy=5e5):
    ent = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    return (ent.item() / np.log(10)) / np.log10(energy / (loss.item() + 1e-7))

# --- 无尽模式引擎 ---
def start_endless_evolution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- POLOAR ENDLESS EVOLUTION START ---")
    
    # 1. 内化世界
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    all_x = torch.stack([train_set[i][0] for i in range(50000)]).to(device)
    all_y = torch.tensor([train_set[i][1] for i in range(50000)]).to(device)
    test_x = torch.stack([test_set[i][0] for i in range(10000)]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(10000)]).to(device)

    # 2. 初始化种群 (10个个体)
    pop_size = 10
    population = [CifarPredatorV3(10**np.random.uniform(-7, -4)).to(device) for _ in range(pop_size)]
    
    gen = 0
    env_noise = 0.1 # 初始环境压力

    while True: # 无尽循环
        print(f"\n[Generation {gen}] Env Noise: {env_noise:.3f}")
        fitness_scores = []
        l_values = []
        
        for i, org in enumerate(population):
            # 每个个体的“猎杀期”
            optimizer = torch.optim.AdamW(org.parameters(), lr=1e-3)
            org.train()
            for _ in range(5): # 每个世代捕食 5 轮
                idx = torch.randperm(50000)[:1024]
                logits, mu, log_var = org(all_x[idx], noise_std=env_noise)
                ce_loss = F.cross_entropy(logits, all_y[idx])
                ent = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
                loss = ce_loss - org.dna * ent
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            # 生存评估
            org.eval()
            with torch.no_grad():
                out, _, _ = org(test_x)
                acc = (out.argmax(1) == test_y).sum().item() / 10000
                out_n, _, _ = org(test_x + 0.4 * torch.randn_like(test_x))
                rob = (out_n.argmax(1) == test_y).sum().item() / 10000
                score = get_fitness(acc, rob)
                L = poloar_l(mu, log_var, ce_loss)
                fitness_scores.append(score)
                l_values.append(L)
                print(f"Org {i} | DNA: {org.dna:.2e} | L: {L:.2f} | Score: {score:.4f}")

        # 自然选择
        indices = np.argsort(fitness_scores)[::-1]
        survivors = [population[i] for i in indices[:5]]
        
        # 进化与变异
        new_population = []
        for parent in survivors:
            new_population.append(parent) # 保留父辈
            offspring = copy.deepcopy(parent)
            # 基因变异
            offspring.dna *= np.random.uniform(0.7, 1.4)
            new_population.append(offspring)
            
        population = new_population
        gen += 1
        env_noise = min(0.3, env_noise + 0.005) # 环境逐渐恶化

        if gen % 10 == 0:
             torch.save(population[0].state_dict(), f"apex_gen_{gen}.pth")

# --- 主程序入口 ---
if __name__ == '__main__':
    print("=== CIFAR-10 Endless Evolution ===")
    print("Starting endless evolution with environmental pressure...")
    start_endless_evolution()