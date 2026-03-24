import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import os

# [物理增强] 开启 CUDNN 极致动员
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- 全局数据预加载 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 预加载MNIST数据到GPU
print("MNIST → GPU...")
train_set = datasets.MNIST(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)

# 预加载数据到GPU
all_x_train = torch.cat([x.unsqueeze(0) for x, _ in train_set], dim=0).to(device)
all_y_train = torch.tensor([y for _, y in train_set], dtype=torch.long).to(device)

all_x_test = torch.cat([x.unsqueeze(0) for x, _ in test_set], dim=0).to(device)
all_y_test = torch.tensor([y for _, y in test_set], dtype=torch.long).to(device)

# 创建数据加载器
train_data = list(zip(all_x_train, all_y_train))
test_data = list(zip(all_x_test, all_y_test))

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

def poloar_diagnostic(mu, log_var, loss, energy_const=1e8):
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
    model = EntropyDrivenModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 使用全局预加载的数据
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512)

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
            # 数据已经在GPU上
            out, _, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
            # 强噪声干扰测试 (0.7 sigma)
            x_noisy = x + 0.7 * torch.randn_like(x)
            out_n, _, _ = model(x_noisy)
            rob_correct += (out_n.argmax(1) == y).sum().item()

    return correct/10000, rob_correct/10000, L

# --- 执行寻找“生命之缝”实验 ---
# 使用二分法逼近 L=1 的 Lambda 值
import numpy as np

# 二分法参数
left = 1e-10
right = 1e-5
tolerance = 1e-12
max_iterations = 50
results = {"λ": [], "acc": [], "rob": [], "L": []}

print(f"{'Iteration':<8} | {'Lambda':<12} | {'L (Lev)':<8} | {'Accuracy':<10} | {'Robustness'}")
print("-" * 65)

for iteration in range(max_iterations):
    lambda_mid = (left + right) / 2
    acc, rob, L = train_cycle(lambda_mid)
    
    results["λ"].append(lambda_mid)
    results["acc"].append(acc)
    results["rob"].append(rob)
    results["L"].append(L)
    
    print(f"{iteration:<8} | {lambda_mid:<12.2e} | {L:<8.3f} | {acc:<10.4f} | {rob:<10.4f}")
    
    # 保存数据到CSV文件（追加模式）- 独立实验文件
    csv_path = r'f:\Entropy_Intell\Code\MNIST\CSV\MNIST_E8.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 检查文件是否存在，不存在则创建表头
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Lambda', 'L', 'Accuracy', 'Robustness'])
        writer.writerow([lambda_mid, L, acc, rob])
    
    # 二分法更新区间
    if abs(L - 1.0) < tolerance:
        print(f"\n找到 L=1 的近似解: λ={lambda_mid:.2e}, L={L:.3f}")
        break
    
    if L < 1.0:
        left = lambda_mid
    else:
        right = lambda_mid

if iteration == max_iterations - 1:
    print(f"\n达到最大迭代次数，最终结果: λ={lambda_mid:.2e}, L={L:.3f}")

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

# 保存图片到指定目录
import os
save_dir = r'f:\Entropy_Intell\Code\MNIST\TB_Figure'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'Life_Point_Search.png'), dpi=300, bbox_inches='tight')
print(f"\n图表已保存到: {os.path.join(save_dir, 'Life_Point_Search.png')}")

# --- 可视化 Lambda-L 关系图 ---
plt.figure(figsize=(10, 5))
plt.plot(results["λ"], results["L"], 'go-', linewidth=2, markersize=8)
plt.xlabel("Lambda (Entropy Driving Force)")
plt.ylabel("Leverage L (Complexity / Budget)")
plt.title("Lambda-L Relationship (POLOAR Phase Diagram)")
plt.axhline(y=1.0, color='r', linestyle='--', label='Life Boundary (L=1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'Lambda_L_Relationship.png'), dpi=300, bbox_inches='tight')
print(f"Lambda-L关系图已保存到: {os.path.join(save_dir, 'Lambda_L_Relationship.png')}")

plt.show()