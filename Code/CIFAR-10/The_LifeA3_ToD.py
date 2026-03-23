import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
from tqdm import tqdm

# --- 1. 宿主 (Host)：引入多巴胺基因 beta ---
class HostOrganismDopa(nn.Module):
    def __init__(self, lambda_dna, beta_dna, latent_dim=256):
        super().__init__()
        self.lambda_dna = lambda_dna # 熵驱动基因 (犹豫度)
        self.beta_dna = beta_dna     # 多巴胺基因 (进取度)
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

# --- 2. 寄生者 (Parasite)：模拟环境压力 ---
class LogicParasite(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(10, 512), nn.GELU(),
            nn.Linear(512, 3072), nn.Tanh()
        )
    def forward(self, labels_onehot):
        noise = self.generator(labels_onehot)
        return noise.view(-1, 3, 32, 32) * 0.2 # 攻击强度 0.2

# --- 3. 物理诊断 ---
def poloar_diagnostic(mu, log_var, loss, energy_const=1e5):
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget

# --- 4. 进化主引擎 ---
def run_dopamine_evolution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"--- DOPAMINE-DRIVEN ECO-SYSTEM START ---")

    # A. 数据全显存载入
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    all_x = torch.stack([train_set[i][0] for i in range(len(train_set))]).to(device)
    all_y = torch.tensor([train_set[i][1] for i in range(len(train_set))]).to(device)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    # B. 初始化种群 (8个宿主，携带 lambda 和 beta 基因)
    hosts = [HostOrganismDopa(lambda_dna=1e-5, beta_dna=0.1).to(device) for _ in range(8)]
    parasites = [LogicParasite().to(device) for _ in range(4)]
    
    gen = 0
    while True:
        host_results = []
        
        # --- 1. 对抗演化阶段 ---
        for h_idx, host in enumerate(hosts):
            parasite = parasites[np.random.randint(len(parasites))]
            opt_h = torch.optim.AdamW(host.parameters(), lr=1e-3)
            opt_p = torch.optim.AdamW(parasite.parameters(), lr=1e-4)
            
            host.train(); parasite.train()
            total_dopa = 0
            
            for epoch_inner in range(5): # 每个世代博弈 5 次
                idx = torch.randperm(50000)[:512]
                x, y = all_x[idx], all_y[idx]
                y_onehot = torch.eye(10, device=device)[y]
                
                # 寄生者攻击
                noise = parasite(y_onehot)
                attacked_x = x + noise
                
                # 宿主反应
                logits, mu, log_var = host(attacked_x)
                ce_loss = F.cross_entropy(logits, y)
                
                # [核心逻辑] 多巴胺计算：预测正确且自信
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    # 多巴胺奖励 = 正确性 * 置信度
                    dopamine_reward = (pred == y).float() * conf
                    batch_dopa = dopamine_reward.mean().item()
                    total_dopa += batch_dopa
                
                # 宿主物理方程：Loss = 痛苦 - λ*灵魂 - β*多巴胺
                entropy_vol = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
                loss_h = ce_loss - host.lambda_dna * entropy_vol - host.beta_dna * batch_dopa
                
                # 寄生者目标：破坏宿主的快感 (让它痛苦)
                loss_p = -ce_loss 
                
                # 梯度博弈
                opt_h.zero_grad(set_to_none=True)
                opt_p.zero_grad(set_to_none=True)
                
                # 计算宿主梯度
                loss_h.backward(retain_graph=True)
                
                # 计算寄生者梯度
                loss_p.backward()
                
                # 更新两个模型参数
                opt_h.step()
                opt_p.step()

            # --- 2. 评估阶段 ---
            host.eval(); parasite.eval()
            with torch.no_grad():
                # 针对攻击生存率
                p_noise = parasite(torch.eye(10, device=device)[test_y])
                out, _, _ = host(test_x + p_noise)
                rob_acc = (out.argmax(1) == test_y).sum().item() / 10000
                
                # 纯净捕食能力
                out_pure, _, _ = host(test_x)
                pure_acc = (out_pure.argmax(1) == test_y).sum().item() / 10000
                
                # 寄生者信息
                noise_norm = p_noise.view(p_noise.size(0), -1).norm(dim=1).mean().item()
                attack_success = (1 - rob_acc) * 100
                
                L = poloar_diagnostic(mu, log_var, ce_loss)
                # 适应度计算：捕食精度 * 生存力 * (1 + 平均多巴胺)
                fitness = pure_acc * rob_acc * (1 + total_dopa/5.0)
                host_results.append({'org': host, 'fit': fitness, 'L': L, 'acc': pure_acc, 'rob': rob_acc, 'dopa': total_dopa/5.0, 'noise_norm': noise_norm, 'attack_success': attack_success})

        # --- 3. 自然选择 ---
        print(f"\n[Generation {gen}] Evolution Report:")
        host_results.sort(key=lambda x: x['fit'], reverse=True)
        
        for i, res in enumerate(host_results[:4]): # 展示前 4 名
            print(f"Rank {i} | DNA: λ={res['org'].lambda_dna:.1e}, β={res['org'].beta_dna:.2f} | L: {res['L']:.2f} | Acc: {res['acc']:.4f} | Dopa: {res['dopa']:.2f} | Fit: {res['fit']:.4f}")
            print(f"      | Parasite: Noise_Norm={res['noise_norm']:.3f}, Attack_Success={res['attack_success']:.1f}%")

        # 末位淘汰与变异
        winners = [res['org'] for res in host_results[:4]]
        new_hosts = []
        for parent in winners:
            new_hosts.append(parent) # 幸存者
            # 后代：λ 和 β 同时变异
            offspring = copy.deepcopy(parent)
            offspring.lambda_dna *= np.random.uniform(0.7, 1.3) # 犹豫基因漂移
            offspring.beta_dna = np.clip(offspring.beta_dna * np.random.uniform(0.8, 1.5), 0.01, 2.0) # 多巴胺基因增强
            new_hosts.append(offspring)
        
        hosts = new_hosts
        gen += 1

if __name__ == '__main__':
    run_dopamine_evolution()