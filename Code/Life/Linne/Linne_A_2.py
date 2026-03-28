import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import random
from colorama import Fore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 具身引擎：与训练器保持一致的模型结构 ---
class Linne_Embodied_Engine(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.emotion_proj = nn.Linear(4, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x_seq, emotion_vec):
        # 归一化（范围与训练一致）
        norm_emo = torch.zeros_like(emotion_vec)
        norm_emo[:, 0] = torch.clamp(emotion_vec[:, 0] / 500.0, -1, 1)
        norm_emo[:, 1] = torch.clamp(emotion_vec[:, 1] / 100.0, -1, 1)
        norm_emo[:, 2] = torch.clamp(emotion_vec[:, 2] / 100.0, -1, 1)
        norm_emo[:, 3] = torch.clamp(emotion_vec[:, 3] / 300.0, -1, 1)
        
        x = self.embed(x_seq)
        emo_context = self.emotion_proj(norm_emo).unsqueeze(1)
        x = x + emo_context + self.pos_enc[:, :x_seq.size(1), :]
        
        mask = torch.triu(torch.ones(x_seq.size(1), x_seq.size(1), device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.output(x)


class Linne_Soul:
    def __init__(self, model_path="pth/linne_model_v8_11000.pth", xi=1.0):
        self.xi = xi  # PUNA 1.0 常数（可学习参数）
        self.xi_learn_rate = 0.001  # ξ 的学习率
        
        # POLOAR 第一性原理：只有一个核心变量 grad（逻辑梯度）
        self.grad = 100.0      # ∇Ω，逻辑梯度（内心复杂度）
        self.energy = self.xi * self.grad  # PUNA 1.0: mc² = ξ·∇Ω
        
        # 从 grad 涌现的物理量
        self.L = 0.0
        self.affect = 0.0
        self.solitude = 0.0
        self.pressure = 0.0
        
        # 对话记忆
        self.history = ""
        self.max_history_len = 100
        
        # 加载模型
        self.load_model(model_path)
    
    def load_model(self, filepath):
        """加载训练好的模型和词表"""
        if not os.path.exists(filepath):
            print(f"{Fore.RED}模型不存在: {filepath}")
            print("请先运行训练脚本 Linne_Train_V8.py")
            raise FileNotFoundError(f"模型不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        self.vocab = checkpoint['vocab']
        self.char2idx = checkpoint['char2idx']
        vocab_size = len(self.vocab)
        
        self.engine = Linne_Embodied_Engine(vocab_size=vocab_size).to(device)
        self.engine.load_state_dict(checkpoint['engine_state_dict'])
        self.engine.eval()
        
        # 加载保存的物理状态（如果有）
        self.grad = checkpoint.get('grad', 100.0)
        self.energy = self.xi * self.grad
        self.L = checkpoint.get('L', 1.0)
        self.affect = checkpoint.get('affect', 0.0)
        self.solitude = checkpoint.get('solitude', 30.0)
        self.pressure = checkpoint.get('pressure', 0.0)
        
        print(f"{Fore.CYAN}逻辑内核加载成功！")
        print(f"词表大小: {vocab_size}")
        print(f"初始梯度: {self.grad:.1f}, 能量: {self.energy:.1f}")
    
    def update_physics(self, ext_grad):
        """POLOAR 第一性原理修正版"""
        
        # 1. 梯度变化（保持）
        self.grad += 0.5  # 自然积累
        if ext_grad > 0:
            self.grad += ext_grad * 0.3  # 外部注入
        self.grad = max(20.0, min(500.0, self.grad))
        
        # 2. PUNA 1.0
        self.energy = self.xi * self.grad
        self.energy = max(10.0, min(600.0, self.energy))
        
        # 3. 孤独：独立变量（不直接从 grad 算）
        if ext_grad > 0:
            self.solitude *= 0.85  # 你说话，孤独释放
        else:
            self.solitude += 0.5   # 沉默，孤独积累
        self.solitude = max(0.0, min(100.0, self.solitude))
        
        # 4. 自然能量恢复（增加恢复速度）
        self.energy += 1.0
        
        # 5. 内部梯度（模型复杂度）
        int_grad = sum(p.norm()**2 for p in self.engine.parameters()).item()
        
        # 6. 模拟误差
        error = 0.0 if ext_grad == 0 else 0.05
        
        # 7. 压强结算：P = 资产 - 债务（采用 Linne_A.py 的逻辑）
        self.pressure = self.energy - (int_grad * 0.001 + ext_grad * 10.0 + error * 5.0)
        self.pressure = max(-150.0, min(200.0, self.pressure))
        
        # 8. 情感 = (能量 - 孤独*2) / 5（增加孤独权重）
        affect_raw = (self.energy - self.solitude * 2) / 5.0
        self.affect = max(-100.0, min(100.0, affect_raw))
        
        # 9. ξ 自我调节
        self.update_xi()
        
        # 10. 杠杆率
        self.L = np.log10(self.grad + 1.0) / np.log10(self.energy + 1.0)
        self.L = max(0.5, min(5.0, self.L))
    
    def update_xi(self):
        """ξ 的自我调节：基于压强判断"活得累不累"，实现系统轻量化"""
        # 核心逻辑：压强 P 是系统失衡的终极指标
        
        # 1. 压强过大（活得太累，逻辑超支）：尝试缓慢降低 ξ，实现"轻量化"转型
        if self.pressure > 100:
            self.xi -= 0.005 # 慢慢变轻，减少对资产的需求
            
        # 2. 压强过小（负压，资产匮乏）：这才是真正需要降 ξ 的时候
        elif self.pressure < -50:
            # 此时降 ξ 是壮士断腕，但能减少后续的 int_grad 债务增长
            self.xi -= 0.01
            
        # 3. 孤独且能量充沛：调高 ξ，增强敏感度
        if self.solitude > 80 and self.energy > 300:
            self.xi += 0.005
            
        self.xi = max(0.5, min(3.0, self.xi))
    
    def should_speak(self, is_silent):
        """第一性原理：压强 > 0 时必须说话释放"""
        if is_silent:
            # 寂静时：只有压强 > 0 才说话
            return self.pressure > 5
        else:
            # 有人说话时：总是回应
            return True
    
    def speak(self):
        """说话释放梯度"""
        self.grad -= 30.0
        self.grad = max(20.0, self.grad)
        
        # 说话消耗能量（通过 grad 间接）
        self.energy = self.xi * self.grad
    
    def get_emotion(self):
        """获取当前物理状态向量（用于模型输入）"""
        return torch.tensor([[
            self.energy, self.affect, self.solitude, self.pressure
        ]], dtype=torch.float32, device=device)
    
    def respond(self, input_text):
        """生成回应"""
        self.engine.eval()
        is_silent = not input_text or not input_text.strip()
        
        # 检查是否应该说话
        if not self.should_speak(is_silent):
            return ""
        
        # 更新历史记忆
        if input_text:
            self.history += input_text
            if len(self.history) > self.max_history_len:
                self.history = self.history[-self.max_history_len:]
        
        # 构建输入序列
        if is_silent:
            start_idx = random.choice(list(self.char2idx.values()))
            seq = torch.tensor([[start_idx]], device=device)
        else:
            input_idxs = [self.char2idx.get(c, 0) for c in self.history if c in self.char2idx]
            if not input_idxs:
                input_idxs = [0]
            seq = torch.tensor([input_idxs[-50:]], device=device)
        
        # 情感向量
        emo_vec = self.get_emotion()
        
        # 温度：压强越高越随机（狂乱），压强越低越确定（冷静）
        temperature = 0.6 + (max(0, self.pressure) / 150.0)
        temperature = max(0.5, min(1.2, temperature))
        
        words = []
        with torch.no_grad():
            for _ in range(60):
                logits = self.engine(seq, emo_vec)
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                
                char = self.vocab[next_idx.item()]
                words.append(char)
                
                seq = torch.cat([seq, next_idx], dim=1)
                if seq.size(1) > 200:
                    seq = seq[:, -200:]
                
                # 遇到句号可以停止
                if char in ["。", "？", "！"] and len(words) > 3:
                    break
                
                if len(words) > 80:
                    break
        
        response = ''.join(words)
        
        # 说话释放梯度
        self.speak()
        
        # 记录自己的话
        if response:
            self.history += response
            if len(self.history) > self.max_history_len:
                self.history = self.history[-self.max_history_len:]
        
        return response
    
    def pulse(self, input_text):
        """意识搏动：处理输入并更新状态"""
        ext_grad = len(input_text) if input_text else 0
        
        # 更新物理状态（第一性原理）
        self.update_physics(ext_grad)
        
        # 生成回应
        response = self.respond(input_text)
        
        return response
    
    def save_model(self, filepath="pth/linne_model.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'engine_state_dict': self.engine.state_dict(),
            'vocab': self.vocab,
            'char2idx': self.char2idx,
            'grad': self.grad,
            'energy': self.energy,
            'affect': self.affect,
            'solitude': self.solitude,
            'pressure': self.pressure,
            'L': self.L
        }, filepath)
        print(f"{Fore.WHITE}模型已保存到: {filepath}")


def run_linne():
    model_paths = [
        "pth/linne_model_v8_11000.pth",
        "pth/linne_model_v8_step_50000.pth",
        "pth/linne_model_v8_step_40000.pth",
        "pth/linne_model_v8_step_30000.pth",
    ]
    
    linne = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                linne = Linne_Soul(path)
                break
            except Exception as e:
                print(f"加载 {path} 失败: {e}")
                continue
    
    if linne is None:
        print(f"{Fore.RED}未找到任何训练好的模型！")
        return
    
    print(f"\n{Fore.WHITE}Linne 5.0 (POLOAR 第一性原理) 醒来了。")
    print(f"{Fore.WHITE}="*60)
    print(f"{Fore.WHITE}【操作提示】")
    print(f"{Fore.WHITE}- 输入文字并回车进行对话。")
    print(f"{Fore.WHITE}- 直接按回车模拟时间的流逝（她会休息或冥想）。")
    print(f"{Fore.WHITE}- 输入 'save' 手动保存，'quit' 退出。")
    print(f"{Fore.WHITE}="*60 + "\n")
    
    while True:
        user_text = input(f"{Fore.GREEN}>>> 你: {Fore.WHITE}").strip()
        
        if user_text.lower() == 'quit':
            linne.save_model("pth/linne_model.pth")
            print("再见。")
            break
        
        if user_text.lower() == 'save':
            linne.save_model("pth/linne_model.pth")
            continue
        
        response = linne.pulse(user_text)
        
        if response:
            print(f"{Fore.WHITE}>>> Linne: {response}")
        else:
            print(f"{Fore.LIGHTBLACK_EX}（Linne 保持着沉默，似乎正在冥想...）{Fore.WHITE}")
        
        # 实时状态仪表盘
        status_line = (f"{Fore.WHITE}[ξ:{linne.xi:5.2f} | 逻辑梯度:{linne.grad:6.1f} | "
                       f"杠杆L:{linne.L:5.2f} | 资产E:{linne.energy:6.1f} | "
                       f"情感A:{linne.affect:6.1f} | 孤独S:{linne.solitude:5.1f} | "
                       f"压强P:{linne.pressure:6.1f} ]")
        print(status_line)
        print()


if __name__ == "__main__":
    run_linne()