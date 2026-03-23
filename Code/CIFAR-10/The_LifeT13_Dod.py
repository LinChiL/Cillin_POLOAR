import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm

# 开启 CUDNN 优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class CifarEntropyLife(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.mu = nn.Linear(4096, latent_dim)
        self.log_var = nn.Linear(4096, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = torch.clamp(self.log_var(feat), -10, 5)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std 
        return self.tail(z), mu, log_var

def poloar_diagnostic_cifar(mu, log_var, loss, energy_const=1e5): 
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget

# --- 核心实验函数 ---
def run_experiment(lambda_val, all_x, all_y, test_x, test_y, epochs=20):
    device = all_x.device
    model = CifarEntropyLife().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    batch_size = 1024
    final_L = 0
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(all_x))
        for i in range(0, len(all_x), batch_size):
            idx = indices[i:i+batch_size]
            x, y = all_x[idx], all_y[idx]
            
            optimizer.zero_grad(set_to_none=True)
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, y)
            entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            loss = ce_loss - lambda_val * entropy
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_L = poloar_diagnostic_cifar(mu, log_var, ce_loss)
            
    # 评估
    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        test_x_noisy = test_x + 0.3 * torch.randn_like(test_x)
        out_n, _, _ = model(test_x_noisy)
        rob = (out_n.argmax(1) == test_y).sum().item() / 10000
        
    return acc, rob, final_L

# --- 二分法主程序 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading data into GPU memory...")
    train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    
    all_x = torch.stack([train_set[i][0] for i in range(len(train_set))]).to(device)
    all_y = torch.tensor([train_set[i][1] for i in range(len(train_set))]).to(device)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    # 二分法参数
    low = 4.90e-5
    high = 5.10e-5
    target_L = 1.0
    tolerance = 0.05 # L在 [0.95, 1.05] 停止
    max_steps = 8

    print(f"\n[Bisection Search] Target L = {target_L}")
    print("-" * 60)
    
    for step in range(max_steps):
        mid = (low + high) / 2
        print(f"Step {step}: Testing λ = {mid:.4e} ...")
        
        acc, rob, L = run_experiment(mid, all_x, all_y, test_x, test_y)
        
        print(f"Result: L = {L:.3f}, Acc = {acc:.4f}, Rob = {rob:.4f}")
        
        if abs(L - target_L) < tolerance:
            print(f"\n🎉 FOUND LIFE POINT! λ = {mid:.4e}")
            break
        
        if L < target_L:
            low = mid # 熵不够，往高了走
        else:
            high = mid # 熵多了，往低了走
            
        torch.cuda.empty_cache() # 释放显存碎片