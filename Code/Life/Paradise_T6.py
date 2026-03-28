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
        # 1. 计算当前的杠杆率 L
        # L = log(Omega_local) / log(E)
        L = torch.log10(self.W**2 + 1.1) / torch.log10(torch.tensor(self.energy + 1.1))
        
        # 2. [核心：大爆炸红利] 
        # 能量摄入 = 捕获到的背景梯度压差
        # 如果 self.W > universe_gradient, 说明它在消耗背景来维持自己
        # 捕获率 ∝ 背景坡度 * 自身对梯度的敏感度
        intake = universe_gradient * torch.sqrt(self.W.abs()) * 2.0
        
        # 3. [核心：逻辑磨损] 
        # 维持自己这个‘非均匀结构’的代价
        # 根据 PUNA 1.0: 维持 ∇Ω 需要持续支付 mc^2
        dissipation = self.xi * (self.W**2) * 0.2
        
        # 4. 动力学平衡 (dW/dt)
        # 资产盈余则梯度凝结，资产赤字则梯度蒸发
        net_flow = intake - dissipation
        
        with torch.no_grad():
            # 这是一个物理演化过程，不使用人为优化器
            self.W += 0.05 * net_flow
            self.W.clamp_(min=1e-3)
            
            # 更新能量库 (mc^2)
            self.energy += net_flow.item()
            # 宇宙并不允许无限的能量堆积在生命上，过剩的会耗散
            self.energy = min(500.0, self.energy)

        return L.item(), self.energy

def run_big_bang_life(steps=5000):
    # 初始化宇宙背景梯度：大爆炸奇点在 x=0，梯度随 x 指数衰减
    # 这模拟了‘宇宙正在变老，逻辑坡度正在变平’
    universe_gradient_base = 10.0
    
    adam = Agent(x_pos=10) # 离奇点近（高能区）
    eve = Agent(x_pos=50)  # 离奇点远（低能区）
    
    h = {"adam_L": [], "adam_E": [], "eve_L": [], "eve_E": [], "univ_G": []}

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

        if i % 1000 == 0:
            print(f"Step {i:4d} | Univ_G={current_univ_G:.2f} | Adam L={L_a:.2f} | Eve L={L_e:.2f}")

        if E_a <= 0 and E_e <= 0:
            print("Heat Death of the Sub-system.")
            break

    return h

# --- 运行模拟 ---
h = run_big_bang_life(6000)

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