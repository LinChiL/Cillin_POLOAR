import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# [物理增强] 开启 CUDNN 极致模式
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class HeavyPredator(nn.Module):
    def __init__(self, latent_dim=1024): # 增加灵魂容量到 1024
        super().__init__()
        # 宽卷积架构：模拟厚重的视觉皮层
        self.sensory_stack = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # 起手就是 128 通道
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # 16x16
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2), # 8x8
            nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # 深度抽象层
            nn.BatchNorm2d(1024), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 反应空间核心 (1024 维度的灵魂)
        self.mu = nn.Linear(1024, latent_dim)
        self.log_var = nn.Linear(1024, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        feat = self.sensory_stack(x)
        mu = self.mu(feat)
        log_var = torch.clamp(self.log_var(feat), -10, 5)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std 
        return self.tail(z), mu, log_var

# --- 物理量度：神级能效预算 ---
def poloar_diagnostic_heavy(mu, log_var, loss, energy_const=1e25): 
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    # 在如此巨大的预算下，L 的波动会变得非常优雅
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget, entropy.item()

# --- 训练循环：全显存内化 ---
def train_heavy_system(lambda_val, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeavyPredator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # 预载 CIFAR-10 到显存
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    full_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    
    print("Inner world loading...")
    all_x = torch.stack([full_set[i][0] for i in range(len(full_set))]).to(device)
    all_y = torch.tensor([full_set[i][1] for i in range(len(full_set))]).to(device)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(all_x))
        batch_size = 256 # 重装模型适当减小 batch
        
        for i in range(0, len(all_x), batch_size):
            idx = indices[i:i+batch_size]
            x, y = all_x[idx], all_y[idx]
            
            # 引入 0.05 的轻微达尔文噪声，训练其防御力
            x = x + 0.05 * torch.randn_like(x)
            
            optimizer.zero_grad(set_to_none=True)
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, y)
            ent_vol = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            
            loss = ce_loss - lambda_val * ent_vol
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            L, E_val = poloar_diagnostic_heavy(mu, log_var, ce_loss)
            if epoch % 5 == 0:
                print(f"Ep {epoch} | L: {L:.3f} | CE: {ce_loss.item():.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        # 强噪声生存测试 (0.4)
        out_n, _, _ = model(test_x + 0.4 * torch.randn_like(test_x))
        rob = (out_n.argmax(1) == test_y).sum().item() / 10000
        
    return acc, rob, L

if __name__ == '__main__':
    # 既然身躯重了，lambda 的杠杆效率会降低，我们可以稍微加大一点
    lambdas = [0, 1e-6, 1e-5, 5e-5, 1e-4]
    
    print(f"\n[HEAVY PREDATOR PROJECT] Launching high-mass evolution...")
    print(f"{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
    print("-" * 60)
    
    for l in lambdas:
        acc, rob, L = train_heavy_system(l)
        print(f"{l:<10.1e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")