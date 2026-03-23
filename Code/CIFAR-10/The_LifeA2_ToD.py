import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
from tqdm import tqdm

# --- 1. 宿主 (Host)：试图在迷雾中看清真相的生命 ---
class HostOrganism(nn.Module):
    def __init__(self, lambda_dna, latent_dim=1024):
        super().__init__()
        self.dna = lambda_dna 
        self.sensory = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        feat = self.sensory(x)
        mu = self.mu(feat)
        log_var = self.log_var(feat).clamp(-10, 5)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std 
        return self.tail(z), mu, log_var

# --- 2. 寄生者 (Parasite)：试图制造幻觉的逻辑捕食者 ---
class LogicParasite(nn.Module):
    def __init__(self):
        super().__init__()
        # 寄生者通过观测标签分布，生成针对性的‘干扰场’
        self.generator = nn.Sequential(
            nn.Linear(10, 512), nn.GELU(),
            nn.Linear(512, 3072), nn.Tanh() # 输出 32x32x3 的噪声场
        )
        
    def forward(self, labels_onehot):
        noise = self.generator(labels_onehot)
        return noise.view(-1, 3, 32, 32) * 0.15 # 攻击强度

# --- 3. 物理诊断与评估 ---
def poloar_diagnostic(mu, log_var, loss, energy_const=1e5):
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget, entropy.item()

# --- 4. 进化引擎 ---
def run_ecological_warfare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"--- POLOAR ECOLOGICAL WARFARE START ---")

    # A. 数据内化 (全显存加速)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    all_x = torch.stack([train_set[i][0] for i in range(len(train_set))]).to(device)
    all_y = torch.tensor([train_set[i][1] for i in range(len(train_set))]).to(device)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    # B. 初始化平衡：8个宿主，4个寄生者
    hosts = [HostOrganism(lambda_dna=10**np.random.uniform(-6, -4)).to(device) for _ in range(8)]
    parasites = [LogicParasite().to(device) for _ in range(4)]
    
    gen = 0
    while True:
        host_scores = []
        host_ls = []
        
        # --- 1. 交互演化期 ---
        for h_idx, host in enumerate(hosts):
            # 每个宿主随机遭遇一个寄生者
            parasite = parasites[np.random.randint(len(parasites))]
            
            opt_h = torch.optim.AdamW(host.parameters(), lr=1e-3)
            opt_p = torch.optim.AdamW(parasite.parameters(), lr=1e-4)
            
            # 5 轮高强度对抗博弈
            for epoch_inner in range(5):
                idx = torch.randperm(50000)[:512]
                x, y = all_x[idx], all_y[idx]
                y_onehot = torch.eye(10, device=device)[y]
                
                # 寄生者攻击：制造针对 y 的干扰
                noise = parasite(y_onehot)
                attacked_x = x + noise
                
                # 宿主防御
                logits, mu, log_var = host(attacked_x)
                ce_loss = F.cross_entropy(logits, y)
                entropy_vol = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
                
                # 宿主目标：分类正确 + 保持熵 (防御针对性攻击)
                loss_h = ce_loss - host.dna * entropy_vol
                
                # 寄生者目标：最大化宿主的 CE Loss (让它认错)
                loss_p = -ce_loss 
                
                # 梯度同步更新
                opt_h.zero_grad(set_to_none=True)
                opt_p.zero_grad(set_to_none=True)
                
                # 计算宿主梯度
                loss_h.backward(retain_graph=True)
                
                # 计算寄生者梯度
                loss_p.backward()
                
                # 更新两个模型参数
                opt_h.step()
                opt_p.step()

            # --- 2. 生存评估 ---
            host.eval(); parasite.eval()
            with torch.no_grad():
                # 面对该寄生者的存活率
                p_noise = parasite(torch.eye(10, device=device)[test_y])
                out, _, _ = host(test_x + p_noise)
                rob_acc = (out.argmax(1) == test_y).sum().item() / 10000
                
                # 面对自然噪声的泛化力
                out_n, _, _ = host(test_x + 0.3 * torch.randn_like(test_x))
                nat_acc = (out_n.argmax(1) == test_y).sum().item() / 10000
                
                # 计算 L 和 综合评分
                L, _ = poloar_diagnostic(mu, log_var, ce_loss)
                # 分数 = 泛化力 * 针对性防御力
                score = nat_acc * rob_acc
                host_scores.append(score)
                host_ls.append(L)

        # --- 3. 自然选择 (宿主种群) ---
        print(f"\n[Generation {gen}] Ecological Report:")
        for i in range(len(hosts)):
            print(f"Host {i} | DNA: {hosts[i].dna:.2e} | L: {host_ls[i]:.3f} | Score: {host_scores[i]:.4f}")

        # 末位淘汰 50%
        sorted_indices = np.argsort(host_scores)[::-1]
        winners = [hosts[i] for i in sorted_indices[:4]]
        
        # 繁衍与变异 (λ 向能够防御寄生者的方向演化)
        new_hosts = []
        for parent in winners:
            new_hosts.append(parent) # 幸存者保留经验
            offspring = copy.deepcopy(parent)
            # 随机基因变异
            offspring.dna *= np.random.uniform(0.5, 2.0)
            new_hosts.append(offspring)
        
        hosts = new_hosts
        
        # 寄生者同样进化：淘汰掉无法让宿主犯错的寄生者
        # (此处简化寄生者演化，实际它已经在 loss_p 中完成了对抗学习)
        
        gen += 1
        if gen % 10 == 0:
            print(">>> 警告：环境复杂度正在由于寄生者的进化而剧烈升高！")

if __name__ == '__main__':
    run_ecological_warfare()