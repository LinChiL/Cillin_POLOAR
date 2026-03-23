import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import numpy as np
import os
from PIL import Image

# [物理环境] 
os.environ['TORCH_HOME'] = r'f:\Entropy_Intell\Code\In_model'

class PredatorSuture(nn.Module):
    """
    POLOAR 缝合插件：赋予死机器‘动态对焦’的能力
    """
    def __init__(self, original_model):
        super().__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = original_model.fc
        
    def diagnose_L(self, mu, log_var, ce_loss, energy_const=1e5):
        entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi)).mean()
        log10_omega = entropy.item() / np.log(10)
        log10_budget = np.log10(energy_const / (ce_loss + 1e-7))
        return log10_omega / log10_budget

    def forward_with_suture(self, x, steps=10, lr=0.1, target_L=1.0):
        device = x.device
        # 1. 提取基础特征 (确定性起点)
        with torch.no_grad():
            feat_base = self.backbone(x).squeeze().detach()
        
        # 2. 将特征设为可学习参数 (这就是‘主观感知的调整’)
        # 我们不改模型，我们只改‘它看到了什么’
        z = feat_base.clone().requires_grad_(True)
        optimizer = torch.optim.SGD([z], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            logits = self.fc(z.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            
            # --- POLOAR 核心优化目标 ---
            # 目标：保持高置信度的同时，最大化输出熵 (防止过早坍缩)
            conf, _ = probs.max(dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            # 这个 Loss 强迫模型在‘寻找答案’和‘保持犹豫’之间平衡
            # 这就是让 L 维持在 1.0 的过程
            suture_loss = - (conf + 0.5 * entropy) 
            
            suture_loss.backward()
            optimizer.step()
            
            # 限制 z 不要偏离原始特征太远 (维持感官真实性)
            with torch.no_grad():
                dist = torch.norm(z - feat_base)
                if dist > 2.0: # 物理边界约束
                    z.copy_(feat_base + (z - feat_base) * (2.0 / dist))

        # 最终产出
        final_logits = self.fc(z.unsqueeze(0))
        return final_logits, z

# --- 运行掠食者缝合实验 ---
def run_suture_optimization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("正在为 ResNet50 缝合‘掠食者之眼’...")
    raw_model = models.resnet50(pretrained=True).to(device)
    raw_model.eval()
    
    suture_system = PredatorSuture(raw_model).to(device)
    
    # 加载 CIFAR-10 数据
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=None)
    img, label = test_set[0]
    
    # 制造极端干扰 (Noise 强度 0.6)
    noisy_array = np.array(img) + np.random.randn(32, 32, 3) * 60
    noisy_img = Image.fromarray(noisy_array.clip(0, 255).astype(np.uint8))
    
    transform = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(noisy_img).unsqueeze(0).to(device)

    print("\n--- POLOAR 缝合优化报告 ---")
    
    # 1. 观测死机器
    with torch.no_grad():
        out_dead = raw_model(x)
        conf_dead, pred_dead = F.softmax(out_dead, dim=1).max(1)
        print(f"【死机器态】: 预测={pred_dead.item()}, 置信度={conf_dead.item():.4f}")

    # 2. 观测缝合优化后的‘掠食者’
    out_suture, z_final = suture_system.forward_with_suture(x, steps=20, lr=0.5)
    conf_suture, pred_suture = F.softmax(out_suture, dim=1).max(1)
    
    # 诊断 L
    mu = z_final.mean()
    log_var = torch.log(z_final.var() + 1e-6)
    ce_loss_final = F.cross_entropy(out_suture, pred_suture).item()
    L_final = suture_system.diagnose_L(mu, log_var, ce_loss_final)

    print(f"【缝合掠食者态】: 预测={pred_suture.item()}, 置信度={conf_suture.item():.4f}, 稳定L={L_final:.3f}")

if __name__ == '__main__':
    run_suture_optimization()