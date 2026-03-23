import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- POLOAR 核心架构 (保持一致性以维持标尺) ---
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

# --- 物理诊断函数 ---
def poloar_diagnostic(mu, log_var, loss, energy_const=1e4):
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    leverage = log10_omega / log10_budget
    return leverage, entropy.item()

# --- 高精度训练循环 ---
def train_singularity_hunt(lambda_entropy, epochs=15):
    device = torch.device("cpu")
    model = EntropyDrivenModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 路径保持你之前的设置
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

    # 评估
    model.eval()
    correct, rob_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            # 强化噪声测试：模拟真实世界的复杂度干扰
            x_noisy = x + 0.7 * torch.randn_like(x)
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    return correct/10000, rob_correct/10000, final_L

# --- 奇点扫描：在 1e-6 到 5e-6 之间精确切片 ---
lambdas = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 8e-6] 
results = {"λ": [], "acc": [], "rob": [], "L": []}

print(f"\n{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
print("-" * 55)

for l in lambdas:
    acc, rob, L = train_singularity_hunt(l)
    results["λ"].append(l)
    results["acc"].append(acc)
    results["rob"].append(rob)
    results["L"].append(L)
    print(f"{l:<10.1e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")

# --- 绘制奇点相图 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Leverage L (Complexity/Budget)')
ax1.set_ylabel('Performance', color='tab:blue')
ax1.plot(results["L"], results["acc"], 'bo-', label='Normal Acc (Logic)', alpha=0.6)
ax1.plot(results["L"], results["rob"], 'ro-', label='Robust Acc (Survival)', linewidth=2)
ax1.axvspan(0.8, 1.2, color='green', alpha=0.2, label='Life Critical Zone')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Lambda Scale', color='tab:gray')
ax2.plot(results["L"], results["λ"], 'k--', alpha=0.3)
ax2.tick_params(axis='y', labelcolor='tab:gray')

plt.title("Experimental Capture of the 'Life Point' (L=1)\nPOLOAR Theory Verification", fontsize=12)
fig.tight_layout()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.show()