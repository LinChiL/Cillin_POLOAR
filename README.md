
[![DOI](https://zenodo.org/badge/1189337876.svg)](https://doi.org/10.5281/zenodo.19206297)
DOI：10.5281/zenodo.19206298

# Cillin_POLOAR

## 项目简介

**Cillin_POLOAR** 是一POLOAR的实验证明实践工程

## 核心理论

### 熵驱动智能假设
智能 = 准确率 × 内部熵

系统存在一个最优熵水平（倒U型曲线），在临界点L=1附近展现出最高的智能表现。

### POLOAR诊断框架
基于物理理论构建的诊断系统，计算：
- **复杂度(Omega)**：系统的内部熵
- **能效预算(Budget)**：系统可用的能量资源
- **杠杆率(Leverage)**：复杂度与预算的比值

杠杆率L是判断系统状态的关键指标：
- **L < 0.8**：过渡态（TRANSITION）
- **0.8 < L < 1.2**：生命态（LIFE）
- **L > 1.2**：坍缩态（COLLAPSED）

## 安装说明

### 依赖库
```bash
pip install torch torchvision matplotlib numpy tqdm
```

### 数据集
项目使用MNIST和CIFAR-10数据集，首次运行时会自动下载到 `Code/In_data/` 目录。

## 使用方法

### MNIST实验
```bash
cd Code/MNIST
python The_LifeT5.py  # 火山口边缘采样
```

### CIFAR-10实验
```bash
cd Code/CIFAR-10
python The_LifeT10_Cama.py  # 优化版本
```

## 实验设置

### 物理参数
- **energy_const**：系统的能量预算，通常设为1e5（CIFAR）或5e3（MNIST）
- **lambda_entropy**：熵的权重系数，控制系统向临界点移动
- **epochs**：训练轮数，通常设为20-25轮

### 诊断指标
- **准确率(Accuracy)**：模型在测试集上的分类准确率
- **鲁棒性(Robustness)**：模型在噪声干扰下的表现
- **杠杆率(Leverage)**：系统状态的关键指标

## 理论背景

本项目基于以下核心假设：
1. 智能系统在临界状态（L≈1）展现出最优性能
2. 熵既是智能的来源，也是系统稳定性的关键
3. 物理诊断框架可以预测系统的演化方向

通过精细调整lambda值，我们寻找能让系统稳定驻留在L=1附近的参数组合，验证熵驱动智能的理论假设。

## 许可证

本项目采用MIT许可证。详情请参阅 [LICENSE.txt](LICENSE.txt) 文件。

## 作者

**Lin**

## 引用

如果您使用了本项目的代码或理论，请引用：
```
@misc{Cillin_POLOAR,
  author = {LinChiL},
  title = {Cillin_POLOAR: Entropy-Driven AI Research},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LinChiL/Cillin_POLOAR}}
}
```
