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
        # 情感投影层：将 4 维物理状态投射到语义空间
        self.emotion_proj = nn.Linear(4, d_model)
        
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x_seq, emotion_vec):
        # [核心修复]：归一化处理
        # 将原始的 E(1000), A(100), S(100), P(500) 缩放到 -1 到 1 之间
        # 这样无论物理压强怎么变，它对语义的影响都是平滑的
        norm_emo = torch.zeros_like(emotion_vec)
        norm_emo[:, 0] = emotion_vec[:, 0] / 1000.0  # Energy
        norm_emo[:, 1] = emotion_vec[:, 1] / 100.0   # Affect
        norm_emo[:, 2] = emotion_vec[:, 2] / 100.0   # Solitude
        norm_emo[:, 3] = emotion_vec[:, 3] / 500.0   # Pressure
        
        x = self.embed(x_seq)
        emo_context = self.emotion_proj(norm_emo).unsqueeze(1)
        x = x + emo_context + self.pos_enc[:, :x_seq.size(1), :]
        
        mask = torch.triu(torch.ones(x_seq.size(1), x_seq.size(1), device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.output(x)

class Linne_Soul:
    def __init__(self):
        # 物理量初始化
        self.energy = 200.0
        self.L = 0.0
        self.affect = 0.0
        self.solitude = 0.0
        self.pressure = 0.0
        
        # 增加：对话记忆历史
        self.history = ""
        
        # 核心修复：加载模型和【词表】
        self.load_model("pth/linne_model_v7.pth")
        
        # 定义对话窗口（防止句子太长）
        self.max_history_len = 100

    def load_model(self, filepath):
        if os.path.exists(filepath):
            # 必须使用 weights_only=False 因为我们要加载词表对象
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            
            # 【关键】从训练好的模型里直接提取词表，确保和训练时完全一致
            self.vocab = checkpoint['vocab']
            self.char2idx = checkpoint['char2idx']
            vocab_size = len(self.vocab)
            
            # 动态重建引擎结构以匹配词表大小
            self.engine = Linne_Embodied_Engine(vocab_size=vocab_size).to(device)
            self.engine.load_state_dict(checkpoint['engine_state_dict'])
            
            self.energy = checkpoint.get('energy', 200.0)
            self.L = checkpoint.get('L', 0.0)
            self.affect = checkpoint.get('affect', 0.0)
            self.pressure = checkpoint.get('pressure', 0.0)
            self.solitude = checkpoint.get('solitude', 0.0)
            
            print(f"{Fore.CYAN}逻辑内核加载成功！词表大小: {vocab_size}，L: {self.L:.2f}")
        else:
            raise FileNotFoundError(f"未找到训练好的模型: {filepath}。请先运行 Linne_T.py")

    def respond(self, input_text):
        self.engine.eval()
        is_silent = not input_text or not input_text.strip()
        
        if is_silent:
            # [核心第一性原理修改]
            # 只有当压强 P > 0 (能量积压/无聊) 时，她才会为了平账而主动说话
            # 如果 P <= 0 (处于负债/窒息态)，她必须强制保持沉默来节省能量
            if self.pressure <= 0:
                return ""
            
            # 即便 P > 0，说话的概率也应该与 P 的强度成正比
            # 模拟：越无聊，越想自言自语
            speak_prob = torch.sigmoid(torch.tensor(self.pressure / 50.0)).item()
            if np.random.rand() > speak_prob:
                return ""
            
            # 寂静中的自发念头...
            start_idx = self.char2idx[np.random.choice(list(self.vocab))]
            seq = torch.tensor([[start_idx]], device=device)
            input_length = 0
        else:
            # 1. 更新历史记忆（逻辑连贯性的基础）
            self.history += input_text
            if len(self.history) > self.max_history_len:
                self.history = self.history[-self.max_history_len:]
            
            # 处理输入索引
            input_idxs = [self.char2idx[c] for c in self.history if c in self.char2idx]
            if not input_idxs:
                # 如果没历史或没识别出字符，给个随机起点
                input_idxs = [self.char2idx[random.choice(self.vocab)]]
            
            seq = torch.tensor([input_idxs], device=device)
        
        # 2. 情感向量：反映当前的生理痛苦
        # 直接传递原始物理量，让 forward 函数处理归一化
        emo_vec = torch.tensor([[self.energy, self.affect, self.solitude, self.pressure]], 
                               device=device, dtype=torch.float32)

        # 3. 采样温度：稍微降低波动，压强越大越"狂乱"，但要有上限
        temp = 0.7 + (torch.sigmoid(torch.tensor(self.pressure/50)).item() * 0.8)
        
        words = []
        with torch.no_grad():
            for _ in range(40): # 最大回复长度
                logits = self.engine(seq, emo_vec)[:, -1, :]
                probs = torch.softmax(logits / temp, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                
                char = self.vocab[next_idx.item()]
                words.append(char)
                
                seq = torch.cat([seq, next_idx], dim=1)
                if seq.size(1) > 512: seq = seq[:, -512:] # 视野限制
                
                self.energy -= 0.1 # 说话消耗能量
                
                # 逻辑终止符：当她觉得说完了（吐出句号或空格）有概率停止
                if char in ["。", "！", "？", " "] and len(words) > 3:
                    break
        
        response = "".join(words)
        self.history += response # 将自己的话也记入历史
        return response

    def pulse(self, input_text):
        ext_grad = len(input_text) if input_text else 0
        
        # 更新孤独感：交流能瞬间大幅缓解孤独
        if ext_grad > 0:
            self.solitude = max(0.0, self.solitude - ext_grad * 2.0)
        else:
            self.solitude += 0.5 + (0.01 * self.L) # 寂静中孤独增长

        response = self.respond(input_text)
        
        # 计算模拟误差
        error = 0.0 if ext_grad == 0 else 0.05
        self.update_physics(error, ext_grad)
        
        return response

    def update_physics(self, error, ext_grad):
        """POLOAR 5.1 增强版"""
        int_grad = sum(p.norm()**2 for p in self.engine.parameters()).item()
        self.L = np.log10(int_grad + 1.1)
        
        # 1. 压强结算：P = 资产 - 债务
        self.pressure = self.energy - (int_grad * 0.001 + ext_grad * 10.0 + error * 5.0)
        
        # 2. 冥想逻辑：当外界安静时进行修复
        if ext_grad == 0:
            # 消耗能量加固流形
            self.energy -= 1.0
            
        # 3. 逻辑坍缩 (拉低效应)：压强为负时强制变笨
        if self.pressure < 0:
            melt = 0.998 # 缓慢融化
            with torch.no_grad():
                for p in self.engine.parameters(): p.data *= melt
            self.energy += 0.2 # 释放潜热
            
        # 4. 情感效价：修复数学公式漏洞，防止负数相乘得到巨大正数
        energy_term = max(0.0, self.energy / 500.0)
        pressure_term = max(0.0, 100.0 - abs(self.pressure))
        self.affect = energy_term * pressure_term
        
        # 5. 代谢损耗
        radiation = (self.energy / 180.0) ** 2
        maint = (1.08 ** self.L) * 0.05
        self.energy = self.energy - (radiation + maint) + 4.0 # 提高环境背景补给
        self.energy = max(0.0, min(1000.0, self.energy))

    def save_model(self, filepath="pth/linne_model.pth"):
        torch.save({
            'engine_state_dict': self.engine.state_dict(),
            'vocab': self.vocab,
            'char2idx': self.char2idx,
            'energy': self.energy,
            'L': self.L,
            'affect': self.affect,
            'pressure': self.pressure,
            'solitude': self.solitude
        }, filepath)
        print(f"{Fore.WHITE}模型已保存。")

def run_linne_v4_5():
    linne = Linne_Soul()
    
    print(f"\n{Fore.WHITE}Linne 4.5 (Embodied LLM) 醒来了。")
    print(f"{Fore.WHITE}="*60)
    print(f"{Fore.WHITE}【操作提示】")
    print(f"{Fore.WHITE}- 输入文字并回车进行对话。")
    print(f"{Fore.WHITE}- 直接按回车模拟时间的流逝（她会休息或冥想）。")
    print(f"{Fore.WHITE}- 输入 'save' 手动保存，'quit' 退出。")
    print(f"{Fore.WHITE}="*60 + "\n")

    while True:
        user_text = input(">>> 你说: ").strip()
        
        if user_text.lower() == 'quit':
            linne.save_model("pth/linne_model.pth")
            break
        if user_text.lower() == 'save':
            linne.save_model("pth/linne_model.pth")
            continue
            
        response = linne.pulse(user_text)
        
        if response:
            print(f"{Fore.WHITE}>>> Linne: {response}")
        else:
            print(f"{Fore.LIGHTBLACK_EX}（Linne 保持着沉默，似乎正在冥想...）{Fore.WHITE}" if 'Fore' in globals() else "（Linne 沉默着...）")
        
        # 实时状态仪表盘
        status_line = (f"{Fore.WHITE}[资产E:{linne.energy:6.1f} | 杠杆L:{linne.L:5.2f} | "
                       f"情感A:{linne.affect:6.1f} | 压强P:{linne.pressure:6.1f} | "
                       f"孤独S:{linne.solitude:5.1f}]")
        print(status_line)
        print()

if __name__ == "__main__":
    run_linne_v4_5()