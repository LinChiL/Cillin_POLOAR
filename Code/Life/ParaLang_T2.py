import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 字母表：a-z + <S> (Stop)
ALPHABET = [chr(i) for i in range(97, 123)] + ["<S>"]
VOCAB_SIZE = len(ALPHABET)

class POLOAR_Brain(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # 视觉编码器：感知苹果坐标
        self.vision_enc = nn.Linear(2, 64)
        self.ln = nn.LayerNorm(64)
        
        # 潜空间 (犹豫中心：产生叠加态)
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)
        
        # 语言生成器 (GRUCell 需要 2D 输入)
        self.rnn_cell = nn.GRUCell(VOCAB_SIZE, latent_dim)
        self.vocal_out = nn.Linear(latent_dim, VOCAB_SIZE)
        
        # 语言接收器 (全序列处理)
        self.listener_rnn = nn.GRU(VOCAB_SIZE, latent_dim, batch_first=True)
        
        # 动作输出
        self.motor_cortex = nn.Linear(latent_dim, 2)

    def forward_speak(self, obs, lambda_ent=0.1):
        """将视觉信号转化为变长字母序列"""
        h = F.gelu(self.ln(self.vision_enc(obs)))
        mu = self.mu(h)
        log_var = torch.clamp(self.log_var(h), -5, 1)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std * lambda_ent
        
        # 递归产生字母
        words_indices = []
        words_soft = [] 
        # curr_h 必须是 [batch, hidden] -> [1, 32]
        curr_h = z 
        curr_input = torch.zeros(1, VOCAB_SIZE, device=z.device)
        
        max_len = 8 # 缩短最大句长，强迫高效沟通
        for _ in range(max_len):
            curr_h = self.rnn_cell(curr_input, curr_h)
            logits = self.vocal_out(curr_h)
            
            # Gumbel-Softmax：离散选择的可微模拟
            soft_token = F.gumbel_softmax(logits, tau=0.8, hard=True)
            idx = torch.argmax(soft_token, dim=1).item()
            
            words_indices.append(idx)
            words_soft.append(soft_token)
            curr_input = soft_token
            
            if ALPHABET[idx] == "<S>": break
            
        entropy = 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi))
        # 返回 [seq_len, VOCAB_SIZE]
        return words_indices, torch.cat(words_soft, dim=0), entropy

    def forward_listen_and_act(self, word_soft):
        """接收信号并执行逻辑坍缩"""
        # word_soft: [seq_len, VOCAB_SIZE] -> [1, seq_len, VOCAB_SIZE]
        input_seq = word_soft.unsqueeze(0)
        _, final_h = self.listener_rnn(input_seq)
        move_goal = self.motor_cortex(final_h.squeeze(0))
        return torch.tanh(move_goal)

class Agent:
    def __init__(self, name):
        self.name = name
        self.brain = POLOAR_Brain()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.003)
        self.reset_physical()

    def reset_physical(self):
        self.pos = torch.rand(2) * 100.0
        self.energy = 500.0
        self.is_alive = True

def run_alphabetic_eden(generations=100):
    adam = Agent("Adam")
    eve = Agent("Eve")
    
    print(f"🌍 POLOAR Alphabetic Eden: No rules, only survival.")

    for gen in range(generations):
        apple_pos = torch.rand(2) * 100.0
        adam.reset_physical()
        eve.reset_physical()
        step = 0
        apples_eaten = 0
        last_word = "..."

        while adam.is_alive and eve.is_alive and step < 800:
            adam.optimizer.zero_grad()
            eve.optimizer.zero_grad()

            # 1. 亚当看苹果，发出‘声码’
            obs_a = (apple_pos - adam.pos).detach() / 50.0
            word_idxs, word_soft, a_ent = adam.brain.forward_speak(obs_a.unsqueeze(0))
            
            # 记录说的话
            spoken_text = "".join([ALPHABET[i] for i in word_idxs if ALPHABET[i] != "<S>"])
            if spoken_text == "": spoken_text = "." # 防止空语
            last_word = spoken_text
            
            # 2. 夏娃听声码，产生位移
            e_move = eve.brain.forward_listen_and_act(word_soft)
            
            # 3. 物理移动
            move_vec = e_move.squeeze() * 6.0
            with torch.no_grad():
                eve.pos = torch.clamp(eve.pos + move_vec, 0, 100)

            # 4. 损失计算
            real_dist = torch.norm(eve.pos - apple_pos)
            survival_pressure = 800.0 / (min(adam.energy, eve.energy) + 1e-6)
            
            # POLOAR 损失：距离 + 压强 - 犹豫奖励
            loss = real_dist + survival_pressure - 0.05 * a_ent
            
            loss.backward()
            adam.optimizer.step()
            eve.optimizer.step()

            # 5. 物理代价：基础损耗 + 犹豫税 + 说话费
            # 每一个字母都是能量凝结的产物
            talk_cost = len(spoken_text) * 0.4 
            maint = 0.8 + a_ent.item() * 0.01 + talk_cost
            adam.energy -= maint
            eve.energy -= maint

            # 6. 成功捕获苹果
            if real_dist.item() < 8.0:
                apples_eaten += 1
                reward = 250.0
                adam.energy += reward * 0.4
                eve.energy += reward * 0.6
                apple_pos = torch.rand(2) * 100.0

            if adam.energy <= 0 or eve.energy <= 0:
                adam.is_alive = False
                eve.is_alive = False
            
            step += 1

        print(f"GEN {gen:2d} | Apples: {apples_eaten:2d} | Last: '{last_word}' | E_A: {adam.energy:.1f}")

        # 自然选择：如果没有产出，进行逻辑突变
        if apples_eaten == 0:
            with torch.no_grad():
                for p in adam.brain.parameters():
                    p.add_(torch.randn_like(p) * 0.05)

if __name__ == "__main__":
    run_alphabetic_eden()