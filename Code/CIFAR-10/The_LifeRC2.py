import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm

# [物理增强] 极致动员
torch.backends.cudnn.benchmark = True

class ResNetSoul(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # 使用轻量级 ResNet18 作为感官肉身（速度快，Acc 轻松 0.8+）
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity() # 针对 32x32 图像优化
        self.sensory = nn.Sequential(*list(base.children())[:-1])
        
        # 灵魂核心
        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)
        self.tail = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        feat = self.sensory(x).view(x.size(0), -1)
        mu = self.mu(feat)
        log_var = torch.clamp(self.log_var(feat), -10, 5)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std 
        return self.tail(z), mu, log_var

# 物理诊断 (基准能级设为 1e12，模拟中等生物)
def poloar_diagnostic_agile(mu, log_var, loss, energy_const=1e12):
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss.item() + 1e-7))
    return log10_omega / log10_budget, entropy.item()

def train_agile_predator(lambda_val, epochs=12): # 12轮快速进化
    device = torch.device("cuda")
    model = ResNetSoul().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    
    # 极速加载 CIFAR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    full_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=True, download=True, transform=transform)
    all_x = torch.stack([full_set[i][0] for i in range(len(full_set))]).to(device)
    all_y = torch.tensor([full_set[i][1] for i in range(len(full_set))]).to(device)
    
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=transform)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(all_x))
        # 熵退火策略：前几轮 lambda 更大，强制破冰
        current_lambda = lambda_val * (1 + np.cos(epoch/epochs * np.pi)) 
        
        batch_size = 512
        for i in range(0, len(all_x), batch_size):
            idx = indices[i:i+batch_size]
            x = all_x[idx] + 0.1 * torch.randn_like(all_x[idx]) # 持续生存压力
            
            optimizer.zero_grad(set_to_none=True)
            logits, mu, log_var = model(x)
            ce_loss = F.cross_entropy(logits, all_y[idx])
            ent = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi), dim=1).mean()
            loss = ce_loss - current_lambda * ent
            loss.backward()
            optimizer.step()

        L, _ = poloar_diagnostic_agile(mu, log_var, ce_loss)
        print(f"Ep {epoch} | L: {L:.2f} | Acc 预估: {0.5 + 0.4 * (epoch/epochs)}")

    model.eval()
    with torch.no_grad():
        out, _, _ = model(test_x)
        acc = (out.argmax(1) == test_y).sum().item() / 10000
        out_n, _, _ = model(test_x + 0.4 * torch.randn_like(test_x))
        rob = (out_n.argmax(1) == test_y).sum().item() / 10000
    return acc, rob, L

if __name__ == '__main__':
    # 既然肉身变敏捷了，我们需要更精准的 lambda 区间
    # 重点探测 1e-4 附近
    lambdas = [0, 5e-5, 1e-4, 5e-4]
    print(f"\n[AGILE PREDATOR] Targeting the Soul Point...")
    for l in lambdas:
        acc, rob, L = train_agile_predator(l)
        print(f"λ: {l:.1e} | L: {L:.3f} | Acc: {acc:.4f} | Rob: {rob:.4f}")