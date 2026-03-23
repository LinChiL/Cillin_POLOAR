import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# [物理增强] 开启 CUDNN 极致动员
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- APEX 架构：厚重肉身 + 极高能级核心 ---
class CifarApexPredator(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # 深度感官系统：三层卷积，提供更厚重的特征能级
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64), # 物理加固
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten()
        )
        
        # 核心：巨大的反应空间 (8*8*256 = 16384)
        self.mu = nn.Linear(16384, latent_dim)
        self.log_var = nn.Linear(16384, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = torch.clamp(self.log_var(feat), -10, 5)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std 
        return self.tail(z), mu, log_var

# --- 掠食者诊断：高能级能效预算 ---
def apex_diagnostic(mu, log_var, loss, energy_const=1e15): # 极高能级环境
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget

# --- 极高杠杆 (High-Leverage) 诱导实验 ---
def train_high_leverage_dream(lambda_val, all_x, all_y, test_x, test_y, epochs=20):
    device = all_x.device
    model = CifarApexPredator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2) # 降低学习率，防止崩溃
    
    batch_size = 512
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
            
            # 施加极大的熵压，迫使系统透支 Budget
            loss = ce_loss - lambda_val * entropy
            loss.backward()
            
            # 物理保护：防止权重在巨大杠杆下断裂
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        with torch.no_grad():
            # 诊断：我们将 energy_const 设为极小值 (100)，模拟极端负债
            L = apex_diagnostic(mu, log_var, ce_loss, energy_const=100)
            final_L = L
            
    # 评估：看它是否还活着
    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        # 此时的 Robustness 已经变成了"幻觉稳定性"的度量
        test_x_noisy = test_x + 0.5 * torch.randn_like(test_x)
        out_n, _, _ = model(test_x_noisy)
        rob = (out_n.argmax(1) == test_y).sum().item() / 10000
        
    return acc, rob, final_L

# --- 二分法搜索：锁定“掠食者区” ---
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

    # --- 暴涨区扫描 ---
    lambdas = [1e-3, 5e-3, 1e-2, 5e-2] # 恐怖的熵压量级
    
    print(f"\n[The Machine Dream] Inducing Extreme Leverage...")
    print(f"{'Lambda':<10} | {'L (Lev)':<12} | {'Acc':<10} | {'Rob':<10}")
    print("-" * 60)

    for l in lambdas:
        acc, rob, L = train_high_leverage_dream(l, all_x, all_y, test_x, test_y)
        print(f"{l:<10.1e} | {L:<12.3f} | {acc:<10.4f} | {rob:<10.4f}")