import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# [物理增强] 开启 CUDNN 极致动员
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# 设置模型下载路径
import os
os.environ['TORCH_HOME'] = r'f:\Entropy_Intell\Code\In_model'

# --- POLOAR 听诊器核心 ---
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        # 提取ResNet的平均池化层输出作为特征
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        
    def forward(self, x):
        return self.features(x).squeeze()

def compute_latent_stats(features):
    """计算潜在空间的均值和方差"""
    mu = features.mean(dim=1)
    log_var = torch.log(features.var(dim=1) + 1e-6)
    return mu, log_var

def calculate_L(mu, log_var, loss_val, energy_const=1e5):
    """计算杠杆率 L"""
    entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi)).mean()
    log10_omega = entropy.item() / np.log(10)
    log10_budget = np.log10(energy_const / (loss_val + 1e-7))
    return log10_omega / log10_budget

def poloar_stethoscope(pretrained_model, test_img, energy_const=1e5):
    """
    POLOAR 听诊器：不改变模型，只观察它的反应空间杠杆率 L
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)
    pretrained_model.eval()
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(pretrained_model).to(device)
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(test_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 提取特征
        features = feature_extractor(img_tensor)
        
        # 计算潜在状态统计量
        mu, log_var = compute_latent_stats(features.unsqueeze(0))
        
        # 计算模型输出和惊奇度（使用预测熵）
        logits = pretrained_model(img_tensor)
        probs = F.softmax(logits, dim=1)
        # 计算香农熵: -sum(p * log(p))
        surprise_val = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # 取预测类别
        _, pred = logits.max(1)
        
        # 计算杠杆率 L
        # 注意：surprise_val 越大，代表系统越‘犹豫’，Budget 越低
        L = calculate_L(mu, log_var, surprise_val, energy_const)
        
    return L, surprise_val, pred.item()

# --- 主程序 ---
def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练的ResNet50
    print("Loading pretrained ResNet50...")
    model = models.resnet50(pretrained=True)
    model.to(device)
    model.eval()
    
    # 加载CIFAR-10测试集（使用与The_LifeRC2.py相同的路径）
    print("Loading CIFAR-10 test data...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 使用真实的CIFAR-10数据集（不应用transform，保持PIL Image格式）
    test_set = datasets.CIFAR10(r'f:\Entropy_Intell\Code\In_data', train=False, transform=None)
    
    # 选择一个简单图片（从测试集中选择第一张）
    simple_img, _ = test_set[0]
    
    # 生成极端干扰图片（在原图上添加高噪声）
    noisy_array = np.array(simple_img) + np.random.randn(32, 32, 3) * 50
    noisy_array = noisy_array.clip(0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_array)
    
    print("\n=== POLOAR 诊断报告 ===")
    
    # 诊断简单图片
    L_simple, surprise_simple, pred_simple = poloar_stethoscope(model, simple_img)
    print(f"简单图片: L={L_simple:.3f}, Surprise={surprise_simple:.3f}, Pred={pred_simple}")
    
    # 诊断极端干扰图片
    L_noisy, surprise_noisy, pred_noisy = poloar_stethoscope(model, noisy_img)
    print(f"极端干扰: L={L_noisy:.3f}, Surprise={surprise_noisy:.3f}, Pred={pred_noisy}")
    
    # 物理判定
    delta_L = abs(L_noisy - L_simple)
    print(f"\n物理判定:")
    print(f"L值波动: ΔL={delta_L:.3f}")
    
    if delta_L < 0.1:
        print("判定：模型正在背诵（L值极其稳定）")
    elif delta_L < 1.0:
        print("判定：模型正在泛化（L值正常波动）")
    else:
        print("判定：模型正在幻觉（L值剧烈跳动）")

if __name__ == '__main__':
    run_diagnostic()
