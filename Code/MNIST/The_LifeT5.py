import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 架构保持不变 (标尺一致性) ---
class QuantumSimLatent(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        log_var = torch.clamp(log_var, -10, 5) 
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std 
        return z, mu, log_var

class EntropyDrivenModel(nn.Module):
    def __init__(self, latent_dim=128, output_dim=10):
        super().__init__()
        self.head = QuantumSimLatent(784, 512, latent_dim)
        self.tail = nn.Linear(latent_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z, mu, log_var = self.head(x)
        logits = self.tail(z)
        return logits, mu, log_var

def poloar_diagnostic(mu, log_var, loss, energy_const=5e3): # 微调基准能效
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    leverage = log10_omega / log10_budget
    return leverage, entropy.item()

def train_nano_scan(lambda_entropy, epochs=20): # 增加轮数以保证临界态自组织
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntropyDrivenModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    data_path = r'f:\Entropy_Intell\Code\In_data'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST(data_path, train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(data_path, train=False, transform=transform), batch_size=128)

    final_L = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'λ={lambda_entropy:.2e} Ep{epoch}', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, y)
            entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            loss = ce_loss - lambda_entropy * entropy
            loss.backward()
            optimizer.step()
            L, _ = poloar_diagnostic(mu, log_var, ce_loss)
            final_L = L
            status = "LIFE" if 0.8 < L < 1.2 else "TRANSITION"
            pbar.set_postfix({'L': f'{L:.3f}', 'Status': status})
    
    # 如果检测到 L 在 0.9-1.1 之间，多跑 5 轮，看看它是否能稳定驻留
    if 0.9 <= final_L <= 1.1:
        print(f"\n⚠️ 检测到 L={final_L:.3f} 在奇点范围内，额外训练 5 轮...")
        for extra_epoch in range(5):
            model.train()
            pbar = tqdm(train_loader, desc=f'λ={lambda_entropy:.2e} ExtraEp{extra_epoch}', leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, mu, log_var = model(x)
                ce_loss = F.cross_entropy(logits, y)
                entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
                loss = ce_loss - lambda_entropy * entropy
                loss.backward()
                optimizer.step()
                L, _ = poloar_diagnostic(mu, log_var, ce_loss)
                final_L = L
                status = "LIFE" if 0.8 < L < 1.2 else "TRANSITION"
                pbar.set_postfix({'L': f'{L:.3f}', 'Status': status})
        print(f"✅ 额外训练完成，最终 L={final_L:.3f}")

    model.eval()
    correct, rob_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            # 维持 0.7 强度，这是“生命”能否在风暴中存活的硬指标
            x_noisy = x + 0.7 * torch.randn_like(x)
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    return correct/10000, rob_correct/10000, final_L

# --- 精确扫描：寻找 L=1.0 的窄门 ---
# 这一次，我们只在这个“火山口”边缘极其精细地移动 
lambdas = [ 
    2.92e-6, 
    2.94e-6, # 理论上的奇点位置 
    2.96e-6, 
    2.98e-6 
 ] 
results = {"λ": [], "acc": [], "rob": [], "L": []}

print(f"\n{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
print("-" * 55)

# 使用进度条包装lambda循环
lambda_pbar = tqdm(lambdas, desc='Nano-scan lambda values', unit='lambda')
for l in lambda_pbar:
    acc, rob, L = train_nano_scan(l)
    results["λ"].append(l)
    results["acc"].append(acc)
    results["rob"].append(rob)
    results["L"].append(L)
    # 使用tqdm的write方法避免与进度条冲突
    lambda_pbar.write(f"{l:<10.1e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")
    
    # 更新进度条信息
    lambda_pbar.set_postfix({
        'Current λ': f'{l:.1e}',
        'L': f'{L:.3f}',
        'Status': 'LIFE' if 0.8 < L < 1.2 else 'TRANSITION'
    })

# ... 绘图部分同上 ...