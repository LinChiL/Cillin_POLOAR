import math

# 常量 (SI单位)
H = 6.62607015e-34
C = 299792458
KB = 1.380649e-23

def analyze_pute(name, m, s, r_obs):
    # 1. 计算总能量 E = mc^2
    energy = m * (C**2)
    log10_e = math.log10(energy)
    
    # 2. 计算地图大小 log10(Ω) = S / (kB * ln10)
    log10_omega = s / (KB * math.log(10))
    
    # 3. 计算物理预算 log10(Budget) = log10(E / (h * R_obs))
    log10_h = math.log10(H)
    log10_r = math.log10(r_obs)
    log10_budget = log10_e - (log10_h + log10_r)
    
    # 4. 计算杠杆率 L = log10(Ω) / log10(Budget)
    leverage = log10_omega / log10_budget if log10_budget != 0 else 0
    
    # 5. 计算观测动员率 log10(η_obs)
    # η = (R * h * Ω) / mc^2 -> log10(η) = log10(R) + log10(h) + log10(Ω) - log10(E)
    log10_eta_obs = log10_r + log10_h + log10_omega - log10_e
    
    return {
        "log10_omega": log10_omega,
        "log10_budget": log10_budget,
        "leverage": leverage,
        "log10_eta_obs": log10_eta_obs
    }

# 扩展数据集
data = [
    # 名称, m(kg), S(J/K), R_obs(Hz)
    ("光子 (1eV)", 1.78e-36, 0, 2.42e14),
    ("单质铁 (Fe)", 9.27e-26, 4.53e-23, 9.79e12),
    ("金刚石 (C)", 1.99e-26, 3.95e-24, 4.65e13),
    ("中子星单元", 1.67e-27, 3.18e-25, 1.00e22),
    ("自复制 RNA", 5.48e-23, 1.33e-20, 1.00e8),
    ("小鼠 N80", 8.00e-10, 2.80e-3, 1.27e1),
    ("普朗克黑洞", 2.17e-8, 8.68e-23, 1.85e43),
    ("太阳质量黑洞", 1.99e30, 6.34e54, 1.00e4),
]

print(f"{'系统名称':<14} | {'log10(Ω)':<12} | {'log10(B)':<10} | {'杠杆率 L':<10} | {'log10(η_obs)'}")
print("-" * 75)

for name, m, s, r in data:
    res = analyze_pute(name, m, s, r)
    
    # 格式化输出，处理极大数据
    omega_str = f"{res['log10_omega']:.2e}" if res['log10_omega'] > 1e6 else f"{res['log10_omega']:.2f}"
    eta_str = f"{res['log10_eta_obs']:.2e}" if abs(res['log10_eta_obs']) > 1e6 else f"{res['log10_eta_obs']:.2f}"
    
    print(f"{name:<14} | {omega_str:>12} | {res['log10_budget']:>10.2f} | {res['leverage']:>10.2f} | {eta_str}")