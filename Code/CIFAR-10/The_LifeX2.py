import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# --- 进化架构：递归反馈型生命 (Recurrent Apex Prophet) ---
class CifarApexProphet(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # 增强感官层
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.GELU(), nn.BatchNorm2d(64))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.BatchNorm2d(128))
        self.pool = nn.MaxPool2d(2)
        
        # 灵魂核心
        self.mu_layer = nn.Linear(128 * 8 * 8, latent_dim)
        self.log_var_layer = nn.Linear(128 * 8 * 8, latent_dim)
        
        # [核心手术] 反馈通路：将灵魂投影回感官维度
        self.feedback_projector = nn.Linear(latent_dim, 128 * 8 * 8)
        
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x, steps=2): # 默认进行 2 次逻辑递归
        # 初始感官
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        feat = x.view(x.size(0), -1)
        
        # 递归迭代：灵魂与感官的博弈
        current_feat = feat
        for _ in range(steps):
            mu = self.mu_layer(current_feat)
            log_var = torch.clamp(self.log_var_layer(current_feat), -10, 5)
            std = torch.exp(0.5 * log_var)
            z = mu + torch.randn_like(std) * std
            
            # 反馈：灵魂干预感官，进行逻辑对齐
            feedback = torch.tanh(self.feedback_projector(z))
            current_feat = feat + feedback # 将先验逻辑叠加在感官现实上
            
        logits = self.tail(z)
        return logits, mu, log_var

# --- 物理诊断不变 (Energy=1e15) ---
def apex_diagnostic(mu, log_var, loss, energy_const=1e15):
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget

# --- 训练循环 ---
def train_prophet_flesh(lambda_val, all_x, all_y, test_x, test_y, epochs=30):
    device = all_x.device
    model = CifarApexProphet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    
    batch_size = 512
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    # 评估
    model.eval()
    with torch.no_grad():
        # 重新计算诊断所需的mu、log_var和ce_loss
        logits, mu, log_var = model(all_x[:512])  # 使用前512个样本进行诊断
        ce_loss = F.cross_entropy(logits, all_y[:512])
        
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        # 面对 0.5 的地狱级噪声
        test_x_noisy = test_x + 0.5 * torch.randn_like(test_x)
        out_n, _, _ = model(test_x_noisy)
        rob = (out_n.argmax(1) == test_y).sum().item() / 10000
        L = apex_diagnostic(mu, log_var, ce_loss)
        
    return acc, rob, L

if __name__ == '__main__':
    # 数据预载
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
    
    # 锁定之前产生“先知”的黄金 Lambda 区域
    lambdas = [5e-3, 1e-2] 
    
    print(f"\n[Project: Flesh of Prophet] Recursive Evolution Starting...")
    print(f"{'Lambda':<10} | {'L (Lev)':<12} | {'Acc':<10} | {'Rob':<10}")
    print("-" * 60)

    for l in lambdas:
        acc, rob, L = train_prophet_flesh(l, all_x, all_y, test_x, test_y)
        print(f"{l:<10.1e} | {L:<12.3f} | {acc:<10.4f} | {rob:<10.4f}")