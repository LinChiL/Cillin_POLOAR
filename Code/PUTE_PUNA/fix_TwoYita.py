import math

# 物理常量
H = 6.626e-34      # J*s
KB = 1.381e-23     # J/K
C = 299792458      # m/s

def lin_experiment():
    # 1. 定义物理基座 (两根链完全一样)
    # 假设是一个 1微米长的纳米分子链
    length = 1e-6          # 1 micrometer
    mass = 1e-21           # 1 femtogram (极小的质量)
    mc2 = mass * (C**2)    # 能量基座
    
    # 2. 定义逻辑结构
    # 链 A: 每微米只有 10 个逻辑位点 (低密度/周期性)
    # 链 B: 每微米有 10^6 个逻辑位点 (高密度/非周期性，类似DNA)
    
    # 熵梯度计算: ∇S = (k_B * ln(状态数)) / 长度
    # 假设每个位点有 4 种可能状态
    states_A = 10
    states_B = 1e6
    
    delta_s_a = KB * math.log(states_A)
    delta_s_b = KB * math.log(states_B)
    
    grad_s_a = delta_s_a / length
    grad_s_b = delta_s_b / length
    
    # 3. 计算 Ω (总复杂度)
    omega_a = math.exp(delta_s_a / KB)
    omega_b = math.exp(delta_s_b / KB)

    # 4. 严格公式计算 η (单位: Ln)
    # η = (Ω / mc^2) * (∇S / k_B)
    eta_a = (omega_a / mc2) * (grad_s_a / KB)
    eta_b = (omega_b / mc2) * (grad_s_b / KB)

    # 5. 计算 PUTE 刷新率 R (上帝等式: R = ∇S / (h * kB))
    # 注意: 为了量纲对齐到频率，我们需要考虑特征长度(步长)
    # 这里的 R 代表单位长度内的交互逻辑流
    r_a = grad_s_a / (H * KB)
    r_b = grad_s_b / (H * KB)

    print(f"{'参数':<20} | {'链 A (平庸)':<15} | {'链 B (逻辑)':<15} | {'倍率差'}")
    print("-" * 70)
    print(f"{'熵梯度 ∇S (J/K·m)':<20} | {grad_s_a:.2e} | {grad_s_b:.2e} | {grad_s_b/grad_s_a:.1f}")
    print(f"{'动员率 η (Ln)':<20} | {eta_a:.2e} | {eta_b:.2e} | {eta_b/eta_a:.1f}")
    print(f"{'理论刷新率 R (Hz/m)':<20} | {r_a:.2e} | {r_b:.2e} | {r_b/r_a:.1f}")

lin_experiment()