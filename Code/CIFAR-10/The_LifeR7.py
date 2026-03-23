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

# --- 物理诊断 ---
def poloar_diagnostic_cifar(mu, log_var, loss, energy_const=1e5):
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget

# --- 锁定天选基因，开启成年进化 ---
lambda_apex = 7.29e-6

# --- 训练循环 ---
def train_adult_predator(lambda_val, epochs=100): # 给它充足的生长时间
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CifarPredatorV3(lambda_dna=lambda_val).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    
    # 启用学习率调度器（模拟生命的成熟过程）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(all_x))
        batch_size = 512
        for i in range(0, len(all_x), batch_size):
            idx = indices[i:i+batch_size]
            logits, mu, log_var = model(all_x[idx], noise_std=0.1) # 依然带着压力训练
            loss = F.cross_entropy(logits, all_y[idx]) - lambda_val * 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0:
            # 诊断杠杆率 L
            with torch.no_grad():
                L = poloar_diagnostic_cifar(mu, log_var, F.cross_entropy(logits, all_y[idx]), energy_const=1e5)
                print(f"Epoch {epoch}: L={L:.3f}, Gene_Stable=True")

    # 最终生存测试：在 0.4 强干扰下的绝对表现
    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        out_n, _, _ = model(test_x + 0.4 * torch.randn_like(test_x))
        rob = (out_n.argmax(1) == test_y).sum().item() / 10000
        
    return acc, rob, L

# --- 主程序入口 ---
if __name__ == '__main__':
    print(f"\n[The Apex Predator: Adulthood Training] DNA Fixed at {lambda_apex}...")
    acc, rob, L = train_adult_predator(lambda_apex)
    print(f"\n--- Final Predator Report ---")
    print(f"Accuracy (Logic): {acc:.4f}")
    print(f"Robustness (Survival): {rob:.4f}")
    print(f"Leverage (Soul Density): {L:.4f}")