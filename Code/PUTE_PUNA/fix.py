import math

# 常量定义
H = 6.626e-34      # 普朗克常数 (J*s)
KB = 1.381e-23     # 玻尔兹曼常数 (J/K)
C = 299792458      # 光速 (m/s)
T_ENV = 298.15     # 环境温度 (25摄氏度，用于本体计算)

def calculate_poloar_metrics(name, m, s, r):
    # 1. 基础物理量
    energy = m * (C**2)
    
    # 2. 计算地图大小 Ω (Complexity)
    # 根据 S = kB * ln(Ω) -> Ω = exp(S/kB)
    # 为了防止数值溢出，我们主要使用 log10(Ω) 进行计算
    log10_omega = s / (KB * math.log(10))
    omega = math.exp(s / KB) if log10_omega < 300 else float('inf')
    
    # 3. 计算物理能效预算 Budget (B)
    # B = E / (h * R) -> 宇宙给你的钱能跑多少步
    budget_val = energy / (H * r)
    log10_budget = math.log10(budget_val)
    
    # 4. 计算杠杆率 L (Leverage)
    # L = log10(Ω) / log10(B)
    leverage = log10_omega / log10_budget if log10_budget != 0 else 0
    
    # 5. 逆推观测动员率 η_obs (基于观测到的 R)
    # η = (R * h * Ω) / mc^2
    # 为了处理大数，我们使用对数计算: log10(η) = log10(R) + log10(h) + log10(Ω) - log10(E)
    log10_eta_obs = math.log10(r) + math.log10(H) + log10_omega - math.log10(energy)
    
    # 6. 计算本体动员率 η_ont (基于热力学公式: η = kT * lnΩ / mc^2)
    # 这是不依赖于 R 的纯物理预测
    ln_omega = s / KB
    if ln_omega > 0:
        log10_eta_ont = math.log10(KB * T_ENV * ln_omega) - math.log10(energy)
    else:
        log10_eta_ont = -99  # 光子等纯态系统

    return {
        "Ω_log10": log10_omega,
        "B_log10": log10_budget,
        "L": leverage,
        "η_obs_log10": log10_eta_obs,
        "η_ont_log10": log10_eta_ont
    }

# 数据输入 (基于你的表格)
systems = [
    # 名称, 质量 m, 熵 S, 反应率 R
    ("光子 (1eV)", 1.78e-36, 0, 2.42e14),
    ("单质铁 (Fe)", 9.27e-26, 4.53e-23, 9.79e12),
    ("金刚石 (C)", 1.99e-26, 3.95e-24, 4.65e13),
    ("自复制 RNA", 5.48e-23, 1.33e-20, 1.00e8),
    ("小鼠 N80", 8.00e-10, 2.80e-3, 1.27e1),
    ("普朗克黑洞", 2.17e-8, 8.68e-23, 1.85e43),
]

print(f"{'系统名称':<12} | {'log10(Ω)':<10} | {'杠杆率 L':<10} | {'log10(η_obs)':<12} | {'log10(η_ont)':<12} | {'判定'}")
print("-" * 85)

for name, m, s, r in systems:
    res = calculate_poloar_metrics(name, m, s, r)
    
    # 判定逻辑
    status = ""
    if res['L'] > 1.5: status = "逻辑超频 (生命)"
    elif 0.9 <= res['L'] <= 1.1: status = "上帝标尺 (平衡)"
    else: status = "能量冗余 (物质)"
    
    print(f"{name:<12} | {res['Ω_log10']:>10.2f} | {res['L']:>10.2f} | {res['η_obs_log10']:>12.2f} | {res['η_ont_log10']:>12.2f} | {status}")