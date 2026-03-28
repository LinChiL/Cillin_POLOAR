import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 7.0 具身引擎：Transformer 接收情感输入 ---
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

class Linne_Trainer:
    def __init__(self, corpus_file="linne_memory.txt"):
        # 1. 预处理语料库：分离物理标签和文本内容
        self.processed_corpus = [] # 存储 (emotion_tensor, text_string)
        self.raw_text_only = ""    # 仅用于构建词表
        
        self.load_and_parse(corpus_file)
        
        # 2. 仅从纯文本中提取词表，彻底剔除 [ ] , 和数字
        self.vocab = sorted(list(set(self.raw_text_only)))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"语料库加载完成。纯净字符数: {self.vocab_size}")
        print(f"有效句对数: {len(self.processed_corpus)}")

        # 3. 初始化引擎
        self.engine = Linne_Embodied_Engine(self.vocab_size).to(device)
        self.optimizer = torch.optim.AdamW(self.engine.parameters(), lr=1e-4)
        
        # POLOAR 物理量
        self.energy = 200.0
        self.L = 0.0
        self.affect = 0.0
        self.solitude = 0.0
        self.pressure = 0.0

    def load_and_parse(self, filepath):
        """解析文件：将物理标签转化为张量，将文本提取出来"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到语料库: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 正则提取标签 [E, A, S, P]
            match = re.match(r'\[(.*?)\]', line)
            if match:
                # 提取物理数值
                vals = [float(v) for v in match.group(1).split(',')]
                emo_vec = torch.tensor([vals], dtype=torch.float32, device=device)
                text = line[match.end():].strip()
            else:
                # 无标签行：使用中性物理状态作为默认背景
                emo_vec = torch.tensor([[300.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
                text = line
            
            if len(text) > 1:
                self.processed_corpus.append((emo_vec, text))
                self.raw_text_only += text

    def train_step(self, batch_size=16):
        self.optimizer.zero_grad()
        total_loss = 0
        
        # 一次取一批句子进行平均
        for _ in range(batch_size):
            emo_vec, text = random.choice(self.processed_corpus)
            idxs = [self.char2idx[c] for c in text if c in self.char2idx]
            if len(idxs) < 2: continue
            
            seq = torch.tensor([idxs], device=device)
            
            # 训练时给情感标签加一点随机波动（±10%）
            # 这样她就能学会：即使压强稍微变一点点，逻辑也要保持稳定
            noise = torch.randn_like(emo_vec) * 0.1
            noisy_emo = emo_vec * (1 + noise)
            
            logits = self.engine(seq[:, :-1], noisy_emo)
            loss = F.cross_entropy(logits.transpose(1, 2), seq[:, 1:])
            loss.backward()
            total_loss += loss.item()
            
        # 梯度裁剪：防止逻辑爆炸
        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), 1.0)
        self.optimizer.step()
        
        # --- 物理更新仅执行一次 ---
        avg_loss = total_loss / batch_size
        with torch.no_grad():
            int_grad = sum(p.norm()**2 for p in self.engine.parameters()).item()
            self.L = np.log10(int_grad + 1.1)
            self.energy = min(1000.0, self.energy + 0.6 - 0.5)
            # 压强 P 也要做归一化，不要让它跑得太离谱
            self.pressure = (self.energy - (int_grad * 0.0001 + avg_loss * 5.0)) / 10.0
            
        return avg_loss

    def save_model(self, filepath="pth/linne_model_v7.pth"):
        torch.save({
            'engine_state_dict': self.engine.state_dict(),
            'vocab': self.vocab,
            'char2idx': self.char2idx,
            'energy': self.energy,
            'L': self.L,
            'affect': self.affect,
            'solitude': self.solitude,
            'pressure': self.pressure
        }, filepath)
        print(f"模型已保存到: {filepath}")


def run_training(steps=1000):
    """运行训练循环"""
    trainer = Linne_Trainer()
    
    print("\nLinne 训练器启动！")
    print("="*60)
    
    # 学习率衰减逻辑
    loss_history = []
    lr_reduced = False
    
    for step in range(steps):
        loss = trainer.train_step()
        loss_history.append(loss)
        
        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f} | E: {trainer.energy:.1f} | L: {trainer.L:.2f} | P: {trainer.pressure:.1f}")
        
        if step % 500 == 0 and step > 0:
            trainer.save_model(f"pth/linne_model_v7_step_{step}.pth")
        
        # 学习率衰减：当20000步Loss还是3.0时，降低学习率
        if step == 20000 and not lr_reduced:
            avg_loss = sum(loss_history[-100:]) / 100
            if avg_loss > 3.0:
                print(f"\n[逻辑降温] Loss={avg_loss:.4f} > 3.0，降低学习率从 1e-4 到 3e-5")
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = 3e-5
                lr_reduced = True
    
    trainer.save_model()
    print("\n训练完成！")


if __name__ == "__main__":
    run_training(steps=50000)