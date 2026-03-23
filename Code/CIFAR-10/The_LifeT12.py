import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# [物理增强] 开启 CUDNN 自动优化，提升动员率 eta
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- POLOAR 进化架构：感官增强型生命 (CNN + Entropy Heart) ---
class CifarEntropyLife(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # 感官系统：卷积层抽象彩色世界
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # 灵魂核心：反应空间 (Reaction Space)
        # 64*8*8 = 4096 是卷积后的特征维度
        self.mu = nn.Linear(4096, latent_dim)
        self.log_var = nn.Linear(4096, latent_dim)
        
        # 决策系统
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = self.log_var(feat)
        log_var = torch.clamp(log_var, -10, 5)
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std 
        
        logits = self.tail(z)
        return logits, mu, log_var

# --- 物理诊断：针对 CIFAR 复杂度提升能效基准 ---
def poloar_diagnostic_cifar(mu, log_var, loss, energy_const=1e5): 
    # CIFAR 需要更高的能量预算来平抑复杂度
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    leverage = log10_omega / log10_budget
    return leverage, entropy.item()

# --- 优化训练循环 ---
def train_evolution_fast(lambda_entropy, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CifarEntropyLife().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 1. 预载数据到内存
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载完整数据集
    full_train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    
    # [关键物理动作] 将 50000 张图直接转为一张巨大的 GPU Tensor
    # 这样训练时就完全不经过 CPU 和磁盘了
    print("put in GPU...")
    all_x = torch.stack([full_train_set[i][0] for i in range(len(full_train_set))]).to(device)
    all_y = torch.tensor([full_train_set[i][1] for i in range(len(full_train_set))]).to(device)
    
    # 测试集同理
    full_test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    test_x = torch.stack([full_test_set[i][0] for i in range(len(full_test_set))]).to(device)
    test_y = torch.tensor([full_test_set[i][1] for i in range(len(full_test_set))]).to(device)

    # 2. 训练循环
    final_L = 0
    for epoch in range(epochs):
        model.train()
        # 使用进度条包装训练循环
        pbar = tqdm(range(0, len(all_x), 1024), desc=f'λ={lambda_entropy:.1e} Ep{epoch}', leave=False)
        
        # 手动打乱索引
        indices = torch.randperm(len(all_x))
        batch_size = 1024
        
        for i in pbar:
            idx = indices[i:i+batch_size]
            x, y = all_x[idx], all_y[idx]
            
            optimizer.zero_grad(set_to_none=True)
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, y)
            entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            loss = ce_loss - lambda_entropy * entropy
            loss.backward()
            optimizer.step()
            
            # 减少诊断频率：每 50 个 batch 诊断一次，显著提升利用率
            if i == len(all_x) - batch_size:
                with torch.no_grad():
                    L, _ = poloar_diagnostic_cifar(mu, log_var, ce_loss)
                    final_L = L
                    status = "LIFE" if 0.8 < L < 1.2 else "TRANSITION"
                    pbar.set_postfix({'L': f'{L:.3f}', 'CE': f'{ce_loss.item():.3f}', 'Status': status})

    # 评估
    model.eval()
    correct, rob_correct = 0, 0
    with torch.no_grad():
        # 直接在显存上评估
        out, _, _ = model(test_x)
        correct += (out.argmax(1) == test_y).sum().item()
        # 真实世界干扰：高强度的色彩抖动噪声
        test_x_noisy = test_x + 0.3 * torch.randn_like(test_x)
        out_n, _, _ = model(test_x_noisy)
        rob_correct += (out_n.argmax(1) == test_y).sum().item()

    return correct/10000, rob_correct/10000, final_L

# --- 在 1e5 能级下捕捉 L=1.0 的生命奇点 ---
# 在 4.5e-5 到 5.0e-5 之间精细扫描
lambdas = [ 
    4.60e-5, # 预估 L 约 -3.5 
    4.75e-5, # 预估 L 约 -1.0 
    4.85e-5, # 理论上的生命奇点 L 约 1.0 
    4.95e-5  # 理论上的过载点 L 约 2.5 
 ]

if __name__ == '__main__':
    print(f"\n[CIFAR-10 Growth Experiment] Sensory-Enhanced Life Simulation...")
    print(f"{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
    print("-" * 55)

    # 使用进度条包装lambda循环
    lambda_pbar = tqdm(lambdas, desc='CIFAR-10 Evolution lambda values', unit='lambda')
    for l in lambda_pbar:
        acc, rob, L = train_evolution_fast(l)
        # 使用tqdm的write方法避免与进度条冲突
        lambda_pbar.write(f"{l:<10.1e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")
        
        # 更新进度条信息
        lambda_pbar.set_postfix({
            'Current λ': f'{l:.1e}',
            'L': f'{L:.3f}',
            'Status': 'LIFE' if 0.8 < L < 1.2 else 'TRANSITION'
        })