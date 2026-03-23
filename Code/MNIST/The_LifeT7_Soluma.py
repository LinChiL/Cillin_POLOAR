import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 架构维持绝对一致性 ---
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

def poloar_diagnostic(mu, log_var, loss, energy_const=5000): # 恢复 5000 基准
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    leverage = log10_omega / log10_budget
    return leverage, entropy.item()

def train_life_sim(lambda_entropy, epochs=20):
    device = torch.device("cpu")
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

    model.eval()
    correct, rob_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            x_noisy = x + 0.7 * torch.randn_like(x)
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    return correct/10000, rob_correct/10000, final_L

# --- 在 5000 能级下寻找生命奇点 ---
# 根据前两次失败的经验，跨越点就在 2.92 到 2.98 之间
lambdas = [2.90e-6, 2.92e-6, 2.94e-6, 2.96e-6, 2.98e-6, 3.00e-6] 
results = {"λ": [], "acc": [], "rob": [], "L": []}

print(f"\n[Energy=5000 Simulation] Starting Singularity Hunt...")
print(f"{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
print("-" * 55)

# 使用进度条包装lambda循环
lambda_pbar = tqdm(lambdas, desc='Singularity Hunt lambda values', unit='lambda')
for l in lambda_pbar:
    acc, rob, L = train_life_sim(l)
    results["λ"].append(l)
    results["acc"].append(acc)
    results["rob"].append(rob)
    results["L"].append(L)
    # 使用tqdm的write方法避免与进度条冲突
    lambda_pbar.write(f"{l:<10.2e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")
    
    # 更新进度条信息
    lambda_pbar.set_postfix({
        'Current λ': f'{l:.2e}',
        'L': f'{L:.3f}',
        'Status': 'LIFE' if 0.8 < L < 1.2 else 'TRANSITION'
    })