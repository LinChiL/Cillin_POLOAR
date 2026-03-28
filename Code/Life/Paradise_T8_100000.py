import torch
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, x_pos, xi=1.0):
        self.x = x_pos  # 在逻辑轴上的位置
        self.xi = xi
        # 内部逻辑梯度 W (代表 L 的物理实体)
        self.W = torch.tensor([1.1], requires_grad=True) 
        self.energy = 100.0 # 初始质能资产

    def survive(self, universe_gradient):
        """
        根据 PUNA 1.0 和 能量守恒 的生存博弈
        """
        # 1. 现状评估
        L = torch.log10(self.W**2 + 1.1) / torch.log10(torch.tensor(self.energy + 1.1))
        
        # 2. 能量收支
        intake = universe_gradient * torch.sqrt(self.W.abs()) * 2.0
        dissipation = self.xi * (self.W**2) * 0.2
        net_flow = intake - dissipation
        
        with torch.no_grad():
            # [核心第一性原理修正：W 的拉低效应]
            # 我们引入一个‘逻辑脆性因子’ (Fragility)
            # 如果 net_flow 是正的，生长是缓慢的（进化需要积累）
            # 如果 net_flow 是负的，坍缩是剧烈的（毁灭只需一瞬）
            
            if net_flow < 0:
                # 能量负债时，W 的下降速度 ∝ 负债程度 / 当前能量
                # 能量越少，W 掉得越快。这叫‘结构性坍缩’
                fragility = 1.0 / (self.energy + 1e-6)
                # 这种强力的拉低效应，保证了系统会在饿死前‘变笨’以保命
                self.W += net_flow * (0.05 + fragility * 10.0)
            else:
                # 能量充裕时，缓慢生长
                self.W += 0.05 * net_flow
            
            self.W.clamp_(min=1e-3)
            
            # 更新能量库
            self.energy += net_flow.item()
            self.energy = min(500.0, self.energy)

        return L.item(), self.energy

def run_big_bang_life(steps=5000):
    # 初始化宇宙背景梯度：大爆炸奇点在 x=0，梯度随 x 指数衰减
    # 这模拟了‘宇宙正在变老，逻辑坡度正在变平’
    universe_gradient_base = 10.0
    
    adam = Agent(x_pos=10) # 离奇点近（高能区）
    eve = Agent(x_pos=50)  # 离奇点远（低能区）
    
    h = {"adam_L": [], "adam_E": [], "eve_L": [], "eve_E": [], "univ_G": []}
    
    last_L = 0.0  # 记录上一步的L值用于计算抖动

    for i in range(steps):
        # 宇宙背景梯度的整体衰减（模拟大爆炸后的膨胀和冷却）
        # 随着 Step 增加，全宇宙的逻辑压差都在变小
        current_univ_G = universe_gradient_base * np.exp(-i / 2000.0)
        
        # 亚当和夏娃在各自的坐标上挣扎
        L_a, E_a = adam.survive(current_univ_G)
        L_e, E_e = eve.survive(current_univ_G * 0.8) # 夏娃在更贫瘠的区域
        
        h["adam_L"].append(L_a)
        h["adam_E"].append(E_a)
        h["eve_L"].append(L_e)
        h["eve_E"].append(E_e)
        h["univ_G"].append(current_univ_G)

        if i % 5000 == 0:
            print(f"Step {i:6d} | L_Adam={L_a:.10f} | Jitter={abs(L_a - last_L):.10e}")
            last_L = L_a

        if E_a <= 0 and E_e <= 0:
            print("Heat Death of the Sub-system.")
            break

    return h

# --- 运行模拟 ---
h = run_big_bang_life(100000)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(h["univ_G"], color='gray', linestyle='--', label="Univ Gradient (Decaying)")
plt.title("Cosmic Background Gradient")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(h["adam_E"], label="Adam (High Energy Zone)")
plt.plot(h["eve_E"], label="Eve (Low Energy Zone)")
plt.title("Energy Assets")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(h["adam_L"], label="Adam L")
plt.plot(h["eve_L"], label="Eve L")
plt.axhline(y=1.0, color='r', linestyle='--')
plt.title("Leverage (Intelligence) Stability")
plt.legend()
plt.show()