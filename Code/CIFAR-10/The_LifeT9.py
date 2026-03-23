import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# --- 进化训练循环 ---
def train_evolution(lambda_entropy, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CifarEntropyLife().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 路径建议改为你存放 CIFAR 的路径
    data_path = r'f:\Entropy_Intell\Code\In_data' 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_loader = DataLoader(datasets.CIFAR10(data_path, train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10(data_path, train=False, transform=transform), batch_size=128)

    final_L = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'λ={lambda_entropy:.1e} Ep{epoch}', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, y)
            entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            loss = ce_loss - lambda_entropy * entropy
            loss.backward()
            optimizer.step()
            L, _ = poloar_diagnostic_cifar(mu, log_var, ce_loss)
            final_L = L
            status = "LIFE" if 0.8 < L < 1.2 else "TRANSITION"
            pbar.set_postfix({'L': f'{L:.3f}', 'CE': f'{ce_loss.item():.3f}', 'Status': status})

    # 评估
    model.eval()
    correct, rob_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            # 真实世界干扰：高强度的色彩抖动噪声
            x_noisy = x + 0.3 * torch.randn_like(x) # CIFAR 0.3 已经很强了
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    return correct/10000, rob_correct/10000, final_L

# --- 在 1e5 能级下寻找 CIFAR 的生命奇点 ---
# 这次的步长必须非常小心，因为 L 的斜率在高维下是近乎垂直的
lambdas = [ 
    2.0e-5, # 预估 L 仍在负数区 
    3.0e-5, # 接近跃迁点 
    3.5e-5, # 理论奇点位置 
    4.0e-5, # 跨越点 
    4.5e-5  # 接近已知的 L=3.5 
 ]

print(f"\n[CIFAR-10 Growth Experiment] Sensory-Enhanced Life Simulation...")
print(f"{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
print("-" * 55)

# 使用进度条包装lambda循环
lambda_pbar = tqdm(lambdas, desc='CIFAR-10 Evolution lambda values', unit='lambda')
for l in lambda_pbar:
    acc, rob, L = train_evolution(l)
    # 使用tqdm的write方法避免与进度条冲突
    lambda_pbar.write(f"{l:<10.1e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")
    
    # 更新进度条信息
    lambda_pbar.set_postfix({
        'Current λ': f'{l:.1e}',
        'L': f'{L:.3f}',
        'Status': 'LIFE' if 0.8 < L < 1.2 else 'TRANSITION'
    })