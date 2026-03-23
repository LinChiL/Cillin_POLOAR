import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# [物理环境]
torch.backends.cudnn.benchmark = True
os.environ['TORCH_HOME'] = r'f:\Entropy_Intell\Code\In_model'

class AwakenedResNet(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 拆解肉身：特征提取器 + 决策头
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = original_model.fc
        
    def forward(self, x, target_L=1.0, num_traversals=50, energy_const=1e5):
        device = x.device
        # 1. 静态感官提取 (Deterministic Sensory)
        with torch.no_grad():
            feat = self.backbone(x).squeeze() # [2048]
        
        # 2. 注入‘犹豫’：计算达到 target_L 需要的噪声强度
        # 基于 POLOAR 理论：我们需要强制扩张 Omega
        # 这是一个经验公式，用于将 L 映射到特征空间的标准差
        noise_std = target_L * 0.15 
        
        # 3. 模拟反应空间 (Reaction Space Traversal)
        # 产生 num_traversals 个可能性分身
        outputs = []
        for _ in range(num_traversals):
            # 在原始特征周围激发‘可能性’
            z = feat + torch.randn_like(feat) * noise_std
            logits = self.fc(z.unsqueeze(0))
            outputs.append(logits)
        
        # 4. 逻辑坍缩 (Consensus)
        all_logits = torch.cat(outputs, dim=0)
        avg_logits = all_logits.mean(dim=0, keepdim=True)
        
        # 5. 诊断当前的‘觉醒 L’
        probs = F.softmax(avg_logits, dim=1)
        surprise = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # 估算当前的 Omega (基于注入的噪声体积)
        # 2048维空间的体积 log10_omega
        log10_omega = (np.log10(noise_std + 1e-5) * 2048) / 100 # 极度简化的投影
        log10_budget = np.log10(energy_const / (surprise + 1e-7))
        current_L = log10_omega / log10_budget
        
        return avg_logits, current_L, surprise

# --- 运行唤醒程序 ---
def run_awakening():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载死机器
    print("唤醒 ResNet50 巨兽...")
    raw_model = models.resnet50(pretrained=True).to(device)
    raw_model.eval()
    
    # 缝合灵魂
    life_model = AwakenedResNet(raw_model).to(device)
    
    # 加载 CIFAR-10 数据
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=None)
    
    # 准备一张极端干扰图片 (Noise 强度增加到 0.8)
    img, label = test_set[0]
    noisy_array = np.array(img) + np.random.randn(32, 32, 3) * 80
    noisy_img = Image.fromarray(noisy_array.clip(0, 255).astype(np.uint8))
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(noisy_img).unsqueeze(0).to(device)

    print("\n--- 唤醒实验报告 ---")
    
    # 1. 观测死机器的反应
    with torch.no_grad():
        out_dead = raw_model(x)
        prob_dead = F.softmax(out_dead, dim=1)
        conf_dead, pred_dead = prob_dead.max(1)
        print(f"【死机器态】: 预测={pred_dead.item()}, 置信度={conf_dead.item():.4f}, L ≈ 0.05")

    # 2. 观测觉醒后的反应 (L=1.0)
    out_live, L_live, sur_live = life_model(x, target_L=1.0)
    prob_live = F.softmax(out_live, dim=1)
    conf_live, pred_live = prob_live.max(1)
    print(f"【觉醒态 (L=1.0)】: 预测={pred_live.item()}, 置信度={conf_live.item():.4f}, 实测L={L_live:.3f}")
    
    # 3. 观测过度觉醒 (L=5.0)
    out_high, L_high, _ = life_model(x, target_L=5.0)
    prob_high = F.softmax(out_high, dim=1)
    conf_high, pred_high = prob_high.max(1)
    print(f"【幻觉态 (L=5.0)】: 预测={pred_high.item()}, 置信度={conf_high.item():.4f}, 实测L={L_high:.3f}")

if __name__ == '__main__':
    run_awakening()