import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy

# [物理增强] 开启 CUDNN 极致动员
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- 掠食者肉身 (同 V2，更紧凑) ---
class CifarOrganism(nn.Module):
    def __init__(self, lambda_val, latent_dim=256):
        super().__init__()
        self.lambda_val = lambda_val # 它的基因
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x, train_mode=True):
        if train_mode: x = x + 0.1 * torch.randn_like(x) # 模拟环境干扰
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = torch.clamp(self.log_var(feat), -10, 5)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std 
        return self.tail(z), mu, log_var

# --- 演化算子 ---
def fitness_function(acc, rob):
    # 掠食者的综合评分：生存力 * 攻击力
    return acc * rob 

def mutate(lambda_val):
    # 基因变异：λ 在对数空间上下波动
    mutation_rate = np.random.uniform(0.5, 2.0)
    return lambda_val * mutation_rate

# --- 种群进化主程序 ---
def run_natural_selection(pop_size=8, generations=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化种群：随机赋予 λ 基因 [1e-6, 1e-4]
    population = [CifarOrganism(lambda_val=10**np.random.uniform(-6, -4)).to(device) for _ in range(pop_size)]
    
    # 预载数据 (内化到显存)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading CIFAR-10 data into GPU memory...")
    train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    
    all_x = torch.stack([train_set[i][0] for i in range(len(train_set))]).to(device)
    all_y = torch.tensor([train_set[i][1] for i in range(len(train_set))]).to(device)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    for gen in range(generations):
        print(f"\n--- Generation {gen} ---")
        scores = []
        
        # 1. 训练与竞争期 (每个世代跑 5 轮)
        for i, org in enumerate(population):
            optimizer = torch.optim.AdamW(org.parameters(), lr=1e-3)
            # 简化训练：每个世代只给有限的“进食时间”
            for _ in range(3): 
                indices = torch.randperm(len(all_x))[:5000] # 随机采样捕食
                logits, mu, log_var = org(all_x[indices], train_mode=True)
                ce_loss = F.cross_entropy(logits, all_y[indices])
                entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
                loss = ce_loss - org.lambda_val * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 2. 评估生存力
            org.eval()
            with torch.no_grad():
                out, _, _ = org(test_x, train_mode=False)
                acc = (out.argmax(1) == test_y).sum().item() / 10000
                out_n, _, _ = org(test_x + 0.4 * torch.randn_like(test_x), train_mode=False)
                rob = (out_n.argmax(1) == test_y).sum().item() / 10000
                score = fitness_function(acc, rob)
                scores.append(score)
                print(f"Org {i}: λ={org.lambda_val:.2e}, Acc={acc:.4f}, Rob={rob:.4f}, Score={score:.4f}")

        # 3. 自然选择：末位淘汰
        sorted_indices = np.argsort(scores)[::-1] # 降序排列
        winners_idx = sorted_indices[:pop_size//2]
        print(f"Survivors: {winners_idx}")
        
        # 4. 繁衍与变异
        new_population = []
        for idx in winners_idx:
            # 幸存者保留
            new_population.append(population[idx])
            # 幸存者克隆并产生变异后代
            offspring = copy.deepcopy(population[idx])
            offspring.lambda_val = mutate(offspring.lambda_val)
            new_population.append(offspring)
            
        population = new_population

    return population[0] # 返回最终的掠食者

# --- 执行 ---
if __name__ == '__main__':
    print("=== CIFAR-10 Natural Selection Evolution ===")
    print("Starting population evolution...")
    
    # 运行自然选择进化
    best_predator = run_natural_selection(pop_size=8, generations=5)
    
    print(f"\n=== Evolution Complete ===")
    print(f"Best Predator: λ={best_predator.lambda_val:.2e}")
    print("The strongest organism has emerged!")