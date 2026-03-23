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
def apex_diagnostic(mu, log_var, loss, energy_const=1e6): # 100万能级
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget

# --- 高能级训练循环 ---
def train_apex_system(lambda_val, all_x, all_y, test_x, test_y, epochs=30):
    device = all_x.device
    model = CifarApexPredator().to(device)
    # 增加重量：引入更强的正则化，防止灵魂溃散
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
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
            loss = ce_loss - lambda_val * entropy
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_L = apex_diagnostic(mu, log_var, ce_loss)
            
    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        # 掠食者测试：面对更残酷的噪声 (0.4 强度)
        test_x_noisy = test_x + 0.4 * torch.randn_like(test_x)
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

    # --- 开始搜索 ---
    # 根据之前 L=2.8 时表现出的强劲溢价，掠食者区应该就在 1e-4 附近
    low = 5e-5
    high = 5e-4 
    
    print(f"\n[Apex Predator Hunt] Searching for the Peak of Evolution...")
    print(f"{'Lambda':<10} | {'L (Lev)':<8} | {'Acc (Logic)':<10} | {'Rob (Survival)':<10} | {'Total Score'}")
    print("-" * 80)

    best_total = 0
    for step in range(8):
        mid = (low + high) / 2
        acc, rob, L = train_apex_system(mid, all_x, all_y, test_x, test_y)
        total = acc + rob
        
        print(f"{mid:<10.2e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f} | {total:.4f}")
        
        if total > best_total:
            best_total = total
            # 掠食者通常出现在 L = 1.5 ~ 4.0 之间
            # 我们通过二分法向更高杠杆靠拢，直到 Acc 开始断崖式跌落
            low = mid
        else:
            high = mid
            
        torch.cuda.empty_cache()