import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class QuantumSimLatent(nn.Module):
    """
    模拟量子叠加态的中间层：
    它不输出一个点，而是一个‘可能性空间’（由均值和协方差定义的分布）
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        # 模拟叠加态的参数：均值和扩展度（Sigma）
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        
        # 重新采样：模拟从叠加态中‘观测’或‘坍缩’的过程
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # 此时系统坍缩到一个具体状态
        return z, mu, log_var

class EntropyDrivenModel(nn.Module):
    def __init__(self, latent_dim=64, output_dim=10):
        super().__init__()
        self.head = QuantumSimLatent(784, 256, latent_dim)
        self.tail = nn.Linear(latent_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        # 经过中间的可能性空间（反应空间）
        z, mu, log_var = self.head(x)
        # 映射到逻辑输出
        logits = self.tail(z)
        return logits, mu, log_var

def compute_reaction_space_volume(mu, log_var):
    """
    计算反应空间论中的‘体积’：
    在经典模拟中，这对应于高维高斯分布的微分熵
    """
    # 简化版：对数行列式的迹（假设独立维度）
    # 实际上代表了系统在犹豫时，涵盖的可能性范围有多广
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1)
    return entropy.mean()

def poloar_diagnostic(mu, log_var, loss, energy_const=1e6):
    """
    基于 PUTE 理论的物理诊断
    """
    # 1. 计算当前系统的复杂度 Omega (熵的指数)
    # 使用 log 空间计算避免数值溢出
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    
    # 2. 计算能效预算 Budget (假设 mc^2 由模型参数量和 Loss 决定)
    # 这里的 energy_const 代表了宇宙分配给这个任务的 '本钱'
    log10_budget = np.log10(energy_const / (loss.item() + 1e-6))
    
    # 3. 计算杠杆率 L
    leverage = log10_omega / log10_budget
    
    return leverage, entropy.item()

def train_entropy_driven(lambda_entropy, epochs=10):
    device = torch.device("cpu")  # 使用CPU避免CUDA兼容性问题
    model = EntropyDrivenModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform), batch_size=128)

    results = {"acc": 0, "rob": 0, "avg_entropy": 0}

    for epoch in range(epochs):
        model.train()
        total_ent = 0
        total_leverage = 0
        
        # 使用进度条包装数据加载器
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit='batch')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, mu, log_var = model(x)
            
            # 1. 逻辑损失 (尾部约束)
            ce_loss = F.cross_entropy(logits, y)
            
            # 2. 熵驱动/反应空间损失 (中间驱动)
            # 我们最大化熵（即最小化负熵），模拟保持相干性，不让系统过快坍缩
            entropy_volume = compute_reaction_space_volume(mu, log_var)
            
            # 核心公式：L = 逻辑确定性 - λ * 可能性空间
            loss = ce_loss - lambda_entropy * entropy_volume
            
            # 3. POLOAR 物理诊断
            leverage, current_ent = poloar_diagnostic(mu, log_var, ce_loss)
            status = "DEAD MATTER" if leverage < 0.1 else "LOGIC/LIFE" if leverage < 1.2 else "DARK MATTER (COLLAPSED)"
            
            loss.backward()
            optimizer.step()
            total_ent += entropy_volume.item()
            total_leverage += leverage
            
            # 更新进度条信息
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Entropy': f'{entropy_volume.item():.4f}',
                'Status': status
            })
        
        # 每轮输出诊断信息
        avg_leverage = total_leverage / len(train_loader)
        avg_entropy = total_ent / len(train_loader)
        print(f"  Epoch {epoch} complete: Leverage={avg_leverage:.4f}, Entropy={avg_entropy:.4f}")

    # 评估：测试泛化与鲁棒性
    model.eval()
    correct = 0
    rob_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # 正常测试
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            # 鲁棒性测试：施加量子环境干扰（加噪）
            x_noisy = x + 0.6 * torch.randn_like(x) 
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    results["acc"] = correct / len(test_loader.dataset)
    results["rob"] = rob_correct / len(test_loader.dataset)
    results["avg_entropy"] = total_ent / len(train_loader)
    return results

# --- 实验运行 ---
lambdas = [0, 1e-4, 5e-4, 1e-3]  # 更细微的区间
acc_list, rob_list, ent_list = [], [], []

print(f"{'Lambda':<8} | {'Accuracy':<10} | {'Robustness':<10} | {'Internal Entropy'}")
print("-" * 50)

# 使用进度条包装lambda循环
lambda_pbar = tqdm(lambdas, desc='Lambda values', unit='lambda')
for l in lambda_pbar:
    print(f"\nRunning experiment with λ = {l}")
    res = train_entropy_driven(l)
    acc_list.append(res["acc"])
    rob_list.append(res["rob"])
    ent_list.append(res["avg_entropy"])
    
    # 更新进度条信息
    lambda_pbar.set_postfix({
        'Current λ': f'{l}',
        'Accuracy': f'{res["acc"]:.4f}',
        'Robustness': f'{res["rob"]:.4f}'
    })
    
    print(f"{l:<8} | {res['acc']:<10.4f} | {res['rob']:<10.4f} | {res['avg_entropy']:.4f}")

# 可视化并保存到指定目录
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lambdas, acc_list, 'bo-', label='Normal Accuracy')
plt.plot(lambdas, rob_list, 'ro-', label='Robust Accuracy')
plt.title("Logic vs. Possibility")
plt.xlabel("Entropy Weight (Lambda)")
plt.ylabel("Performance")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambdas, ent_list, 'go-')
plt.title("Internal 'Reaction Space' Volume")
plt.xlabel("Entropy Weight")
plt.ylabel("Entropy")

# 保存图表到指定目录
plt.tight_layout()
plt.savefig(r'f:\Entropy_Intell\Code\figure\Logic_Entropy.png', dpi=150)
plt.close()