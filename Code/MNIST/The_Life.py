import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- POLOAR 核心模型 ---
class QuantumSimLatent(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) # 能量归一化：维持 mc^2 的稳定性
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        # 限制 log_var 防止熵爆炸 (锁死在物理极限内)
        log_var = torch.clamp(log_var, -10, 5) 
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std 
        return z, mu, log_var

class EntropyDrivenModel(nn.Module):
    def __init__(self, latent_dim=128, output_dim=10): # 增加潜变量维度以提升能效预算
        super().__init__()
        self.head = QuantumSimLatent(784, 512, latent_dim)
        self.tail = nn.Linear(latent_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z, mu, log_var = self.head(x)
        logits = self.tail(z)
        return logits, mu, log_var

# --- 物理量度 ---
def compute_reaction_space_volume(mu, log_var):
    # 计算微分熵：代表系统的反应空间体积 Omega
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1)
    return entropy.mean()

def poloar_diagnostic(mu, log_var, loss, energy_const=1e4):
    """
    POLOAR 诊断系统：计算杠杆率 L
    注意：调整了 energy_const 以适应当前模型规模
    """
    entropy = compute_reaction_space_volume(mu, log_var)
    log10_omega = entropy.item() / np.log(10)
    
    # 预算 Budget：由模型资产和逻辑损失决定
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    
    leverage = log10_omega / log10_budget
    return leverage, entropy.item()

# --- 训练逻辑 ---
def train_cycle(lambda_entropy, epochs=8):
    device = torch.device("cpu")
    model = EntropyDrivenModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform), batch_size=128)

    history = {"L": [], "E": []}

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'λ={lambda_entropy:.1e} Ep{epoch}', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, y)
            ent_vol = compute_reaction_space_volume(mu, log_var)
            
            # 核心方程：损失 = 逻辑 - λ * 熵
            loss = ce_loss - lambda_entropy * ent_vol
            
            # 物理诊断
            L, cur_ent = poloar_diagnostic(mu, log_var, ce_loss)
            status = "DEAD" if L < 0.2 else "LIFE" if L < 1.3 else "DARK"
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'L': f'{L:.2f}', 'Status': status, 'CE': f'{ce_loss.item():.3f}'})

    # 最终评估
    model.eval()
    correct, rob_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            # 强噪声干扰测试 (0.7 sigma)
            x_noisy = x + 0.7 * torch.randn_like(x)
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    return correct/10000, rob_correct/10000, L

# --- 执行寻找“生命之缝”实验 ---
# 极微量级的 lambda 探索
lambdas = [0, 1e-7, 1e-6, 5e-6, 1e-5, 5e-5] 
results = {"λ": [], "acc": [], "rob": [], "L": []}

print(f"{'Lambda':<10} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
print("-" * 55)

for l in lambdas:
    acc, rob, L = train_cycle(l)
    results["λ"].append(l)
    results["acc"].append(acc)
    results["rob"].append(rob)
    results["L"].append(L)
    print(f"{l:<10.1e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")

# --- 可视化倒U型曲线 ---
plt.figure(figsize=(10, 5))
plt.plot(results["L"], results["acc"], 'bo-', label='Normal Acc (Logic)')
plt.plot(results["L"], results["rob"], 'ro-', label='Robust Acc (Survival)')
plt.axvline(x=1.0, color='g', linestyle='--', label='Life Point (L=1)')
plt.xlabel("Leverage L (Complexity / Budget)")
plt.ylabel("Performance")
plt.title("The Search for the Life-Point (POLOAR Theory)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()