import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 具身引擎：Transformer 接收情感输入 ---
class Linne_Embodied_Engine(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.emotion_proj = nn.Linear(4, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model))
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=1024, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x_seq, emotion_vec):
        # 归一化物理量
        norm_emo = torch.zeros_like(emotion_vec)
        norm_emo[:, 0] = torch.clamp(emotion_vec[:, 0] / 500.0, -1, 1)   # Energy
        norm_emo[:, 1] = torch.clamp(emotion_vec[:, 1] / 100.0, -1, 1)   # Affect
        norm_emo[:, 2] = torch.clamp(emotion_vec[:, 2] / 100.0, -1, 1)   # Solitude
        norm_emo[:, 3] = torch.clamp(emotion_vec[:, 3] / 300.0, -1, 1)   # Pressure
        
        x = self.embed(x_seq)
        emo_context = self.emotion_proj(norm_emo).unsqueeze(1)
        x = x + emo_context + self.pos_enc[:, :x_seq.size(1), :]
        
        mask = torch.triu(torch.ones(x_seq.size(1), x_seq.size(1), device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.output(x)


class Linne_Trainer:
    def __init__(self, corpus_file="linne_memory.txt"):
        # 1. 加载语料库
        self.processed_corpus = []  # 存储 (emotion_tensor, text)
        self.raw_text_only = ""
        self.load_and_parse(corpus_file)
        
        # 2. 构建词表
        self.vocab = sorted(list(set(self.raw_text_only)))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"语料库加载完成。独立字符数: {self.vocab_size}")
        print(f"有效句对数: {len(self.processed_corpus)}")
        
        # 3. 初始化引擎
        self.engine = Linne_Embodied_Engine(self.vocab_size).to(device)
        self.optimizer = torch.optim.AdamW(self.engine.parameters(), lr=1e-4)
        
        # 4. 物理状态（将向语料库标签演化）
        self.energy = 200.0
        self.affect = 0.0
        self.solitude = 0.0
        self.pressure = 0.0
        self.L = 1.0

    def load_and_parse(self, filepath):
        """解析语料库，提取 [E,A,S,P] 标签和文本"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"语料库不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'\[(.*?)\]', line)
            if match:
                vals = [float(v.strip()) for v in match.group(1).split(',')]
                if len(vals) == 4:
                    emo_vec = torch.tensor([vals], dtype=torch.float32, device=device)
                    text = line[match.end():].strip()
                else:
                    continue
            else:
                # 无标签行：中性状态
                emo_vec = torch.tensor([[200.0, 0.0, 30.0, 0.0]], dtype=torch.float32, device=device)
                text = line
            
            if len(text) >= 2:
                self.processed_corpus.append((emo_vec, text))
                self.raw_text_only += text
    
    def get_current_emotion(self):
        """获取当前物理状态向量"""
        return torch.tensor([[
            self.energy, self.affect, self.solitude, self.pressure
        ]], dtype=torch.float32, device=device)
    
    def update_physics_to_target(self, target_emo, alpha=0.1):
        """让物理状态向目标标签平滑演化"""
        with torch.no_grad():
            self.energy = self.energy * (1 - alpha) + target_emo[0, 0].item() * alpha
            self.affect = self.affect * (1 - alpha) + target_emo[0, 1].item() * alpha
            self.solitude = self.solitude * (1 - alpha) + target_emo[0, 2].item() * alpha
            self.pressure = self.pressure * (1 - alpha) + target_emo[0, 3].item() * alpha
            
            # 限制范围
            self.energy = max(10.0, min(600.0, self.energy))
            self.affect = max(-100.0, min(100.0, self.affect))
            self.solitude = max(0.0, min(100.0, self.solitude))
            self.pressure = max(-200.0, min(300.0, self.pressure))
    
    def train_step(self, batch_size=8):
        """训练一步：随机抽取一批数据，用当前物理状态训练"""
        self.engine.train()
        self.optimizer.zero_grad()
        total_loss = 0
        valid_count = 0
        
        for _ in range(batch_size):
            # 随机抽取一条语料
            target_emo, text = random.choice(self.processed_corpus)
            
            # 物理状态向目标标签演化（让她“体验”这个状态）
            self.update_physics_to_target(target_emo, alpha=0.05)
            
            # 编码文本
            idxs = [self.char2idx.get(c, None) for c in text]
            idxs = [i for i in idxs if i is not None]
            if len(idxs) < 2:
                continue
            
            seq = torch.tensor([idxs], device=device)
            current_emo = self.get_current_emotion()
            
            # 前向传播
            logits = self.engine(seq[:, :-1], current_emo)
            loss = F.cross_entropy(logits.transpose(1, 2), seq[:, 1:])
            loss = loss / batch_size
            loss.backward()
            total_loss += loss.item() * batch_size
            valid_count += 1
        
        if valid_count == 0:
            return 0.0
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新 L（杠杆率）
        with torch.no_grad():
            int_grad = sum(p.norm().item()**2 for p in self.engine.parameters())
            self.L = np.log10(int_grad + 1.1) * (self.energy / 500.0)
            self.L = max(0.5, min(8.0, self.L))
        
        return total_loss / valid_count
    
    def save_model(self, filepath="pth/linne_model_v8.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'engine_state_dict': self.engine.state_dict(),
            'vocab': self.vocab,
            'char2idx': self.char2idx,
            'energy': self.energy,
            'affect': self.affect,
            'solitude': self.solitude,
            'pressure': self.pressure,
            'L': self.L
        }, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath="pth/linne_model_v8.pth"):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            self.engine.load_state_dict(checkpoint['engine_state_dict'])
            self.vocab = checkpoint['vocab']
            self.char2idx = checkpoint['char2idx']
            self.energy = checkpoint.get('energy', 200.0)
            self.affect = checkpoint.get('affect', 0.0)
            self.solitude = checkpoint.get('solitude', 0.0)
            self.pressure = checkpoint.get('pressure', 0.0)
            self.L = checkpoint.get('L', 1.0)
            print(f"模型已加载: {filepath}")
        else:
            print(f"模型不存在，从头开始: {filepath}")


def run_training(steps=50000, save_interval=500):
    trainer = Linne_Trainer("linne_memory.txt")
    
    print("\n" + "="*60)
    print("Linne 训练器 V8 启动")
    print("="*60)
    print("物理状态将从语料库标签中学习")
    print("="*60 + "\n")
    
    loss_history = []
    
    for step in range(steps):
        loss = trainer.train_step(batch_size=8)
        loss_history.append(loss)
        
        if step % 100 == 0:
            avg_loss = sum(loss_history[-100:]) / 100 if loss_history else loss
            print(f"Step {step:6d} | Loss: {loss:.4f} | Avg: {avg_loss:.4f} | "
                  f"E:{trainer.energy:.1f} A:{trainer.affect:.1f} S:{trainer.solitude:.1f} P:{trainer.pressure:.1f} | L:{trainer.L:.2f}")
        
        if step > 0 and step % save_interval == 0:
            trainer.save_model(f"pth/linne_model_v8_step_{step}.pth")
        
        # 学习率衰减
        if step == 15000:
            avg = sum(loss_history[-500:]) / 500
            if avg > 2.5:
                for pg in trainer.optimizer.param_groups:
                    pg['lr'] = 5e-5
                print(f"\n[学习率衰减] lr=5e-5, avg_loss={avg:.4f}\n")
        
        if step == 30000:
            avg = sum(loss_history[-500:]) / 500
            if avg > 1.5:
                for pg in trainer.optimizer.param_groups:
                    pg['lr'] = 2e-5
                print(f"\n[学习率衰减] lr=2e-5, avg_loss={avg:.4f}\n")
    
    trainer.save_model("pth/linne_model_v8_final.pth")
    print("\n训练完成！")


if __name__ == "__main__":
    run_training(steps=50000, save_interval=1000)