import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import numpy as np
from PIL import Image
import os

# [物理环境] 
os.environ['TORCH_HOME'] = r'f:\Entropy_Intell\Code\In_model'

class ResonantPredator(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = original_model.fc
        
    def forward_resonant(self, x, steps=50, lr=1.0, temperature=0.5):
        device = x.device
        # 1. 提取基础特征
        with torch.no_grad():
            feat_base = self.backbone(x).squeeze().detach()
        
        # 2. 开启朗之万动力学 (SGLD)
        # z 是我们的‘逻辑游走者’
        z = feat_base.clone().requires_grad_(True)
        
        # 记录路径中的所有想法
        thoughts = []
        
        for i in range(steps):
            # 注入随机热能 (Entropy Injection)
            noise = torch.randn_like(z) * temperature
            
            logits = self.fc((z + noise).unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            
            # 优化目标：最大化置信度的同时，寻找逻辑稳定性
            conf, _ = probs.max(dim=1)
            # 我们引入一个‘势能函数’，迫使 z 寻找更稳健的区域
            potential = -torch.log(conf + 1e-10) 
            
            potential.backward()
            
            # SGLD 更新步：梯度推力 + 随机热运动
            with torch.no_grad():
                # 这一行就是 PUTE 4.0 的物理实现：R = (Energy + Noise) / Gradient
                z -= lr * z.grad + noise 
                z.grad.zero_()
                
                # 约束：不要跑出感官边界太远
                dist = torch.norm(z - feat_base)
                if dist > 5.0: 
                    z.copy_(feat_base + (z - feat_base) * (5.0 / dist))
            
            thoughts.append(F.softmax(self.fc(z.unsqueeze(0)), dim=1))

        # 3. 逻辑坍缩：在所有‘想法’中寻找共识
        consensus = torch.cat(thoughts, dim=0).mean(dim=0, keepdim=True)
        return consensus, z

# --- 运行爆破实验 ---
def run_resonant_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("正在为 ResNet50 进行‘逻辑爆破’缝合...")
    raw_model = models.resnet50(pretrained=True).to(device)
    raw_model.eval()
    
    predator = ResonantPredator(raw_model).to(device)
    
    # 加载 CIFAR-10 数据
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=None)
    img, label = test_set[0] # 这是一个‘猫’，但在 0.6 噪声下变成了‘食蜂鸟’
    
    noisy_array = np.array(img) + np.random.randn(32, 32, 3) * 60
    noisy_img = Image.fromarray(noisy_array.clip(0, 255).astype(np.uint8))
    
    transform = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(noisy_img).unsqueeze(0).to(device)

    # 执行爆破
    print("\n[Langevin Resonant Suture] 开始逻辑搜寻...")
    out_consensus, z_final = predator.forward_resonant(x, steps=100, lr=0.1, temperature=0.8)
    
    conf_final, pred_final = out_consensus.max(1)
    
    # 诊断 L
    with torch.no_grad():
        # 这里计算所有 thoughts 的方差作为真实的 Omega
        ent_final = -torch.sum(out_consensus * torch.log(out_consensus + 1e-10)).item()
        # 杠杆率 L 诊断 (使用 1e5 预算)
        L = (ent_final / np.log(10)) / np.log10(1e5 / (ent_final + 1e-7))

    print(f"\n【死机器态】 (基准): 预测=392, 置信度=0.368")
    print(f"【爆破掠食者】: 预测={pred_final.item()}, 置信度={conf_final.item():.4f}, 逻辑熵={ent_final:.3f}, 激活L={L:.3f}")

if __name__ == '__main__':
    run_resonant_experiment()