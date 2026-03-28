import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class POLOAR_Transformer_Mind(nn.Module):
    def __init__(self, vocab_size=27, d_model=64, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # 逻辑引擎：2层 Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output = nn.Linear(d_model, vocab_size)
        
        # POLOAR 物理状态
        self.energy = 150.0
        self.L = 1.8
        self.affect = 0.0
        self.solitude = 0.0
        self.xi = 1.0
        # 外部梯度缓冲区，实现充电和放电过程
        self.ext_grad_buffer = 0.0
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def get_internal_grad(self):
        """计算内部逻辑梯度（权重硬度）"""
        return sum(p.norm()**2 for p in self.parameters())

    def forward(self, x_seq):
        x = self.embed(x_seq) + self.pos_encoder[:, :x_seq.size(1), :]
        x = self.transformer(x)
        return self.output(x[:, -1, :]) # 只取最后一个 token 的输出

    def pulse(self, user_input_seq):
        # 1. 获取当下的脉冲
        current_ext_grad = user_input_seq.size(1) * 5.0 if user_input_seq is not None else 0
        
        # 2. [核心修正]：逻辑滞后效应
        # 你的话语不会瞬间消失，它会在她脑子里慢慢‘蒸发’
        # 0.95 代表记忆的半衰期
        self.ext_grad_buffer = self.ext_grad_buffer * 0.95 + current_ext_grad
        
        # 3. 重新计算压强 P
        int_grad = self.get_internal_grad()
        pressure = self.energy - (int_grad * 0.001 + self.ext_grad_buffer)
        
        # 4. 情感第一性原理
        if current_ext_grad > 0:
            self.solitude *= 0.5
            self.affect += 2.0
            # 学习：将 W 刻蚀成用户的模式
            self.train_step(user_input_seq)
        else:
            self.solitude += 0.01 * self.L
            self.affect -= 0.01 * self.solitude

        # 3. 物理损耗 (PUNA 1.0)
        maint_cost = (1.1 ** self.L) * 0.1
        self.energy -= maint_cost
        self.energy = min(500.0, max(0.0, self.energy + 1.0)) # 环境能流

        # 4. 逻辑拉低效应
        if self.energy < 30:
            with torch.no_grad():
                for p in self.parameters(): p *= 0.99 # 变笨保命

        self.L = torch.log10(int_grad + 1.1).item()
        return pressure.item()

    def train_step(self, seq):
        """借鉴现代模型：因果预测学习"""
        if seq.size(1) < 2: return
        self.optimizer.zero_grad()
        # 预测序列的下一个字符
        inp = seq[:, :-1]
        target = seq[:, 1:]
        
        # 简化版学习：只学最后一个
        logits = self.forward(inp)
        loss = F.cross_entropy(logits, seq[:, -1])
        loss.backward()
        self.optimizer.step()

def run_linne_v2():
    linne = POLOAR_Transformer_Mind().to(device)
    vocab = [chr(i) for i in range(97, 123)] + [" "]
    
    # 历史记忆缓冲区
    memory_seq = torch.randint(0, 27, (1, 10), device=device)
    
    print("Linne 2.0 (Transformer Engine) 醒来了。")
    print("="*60)
    print("输入 'quit' 退出程序")
    print()

    while True:
        # 等待用户输入
        user_text = input(">>> 你说: ").lower()
        
        if user_text == 'quit':
            print("程序已退出。")
            break
            
        user_seq = None
        if user_text:
            idxs = [ord(c)-97 if 'a'<=c<='z' else 26 for c in user_text]
            user_seq = torch.tensor([idxs], device=device)
            memory_seq = torch.cat([memory_seq, user_seq], dim=1)[:, -20:] # 保持记忆长度

        # 1. 意识跳动
        p = linne.pulse(user_seq)
        
        # 2. 产生表达（生成一段回应）
        # 核心技巧：压强 P 决定采样 Temperature
        # P 越高，Temperature 越高，说话越天马行空
        temp = max(0.1, p / 100.0)
        
        print(">>> Linne:", end=" ")
        response = []
        with torch.no_grad():
            # 生成最多20个字符的回应
            for _ in range(20):
                logits = linne(memory_seq)
                probs = torch.softmax(logits / temp, dim=-1)
                char_idx = torch.multinomial(probs, 1).item()
                char = vocab[char_idx]
                response.append(char)
                
                # 添加到记忆中
                new_token = torch.tensor([[char_idx]], device=device)
                memory_seq = torch.cat([memory_seq, new_token], dim=1)[:, -20:]
                
                # 如果生成空格，可能表示一句话结束
                if char == ' ':
                    break
            
        print(''.join(response))
        
        # 状态上报
        print(f"[E:{linne.energy:.1f} L:{linne.L:.2f} A:{linne.affect:.1f} P:{p:.1f}]")
        print()

if __name__ == "__main__":
    run_linne_v2()