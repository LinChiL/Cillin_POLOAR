import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 增加逻辑红利，降低思维税，给数学演化留出资产空间
ENERGY_REWARD = 1000.0 
MAINT_BASE = 0.2

class POLOAR_Brain(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.vision_enc = nn.Linear(2, 64)
        self.ln = nn.LayerNorm(64)
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)
        self.rnn_cell = nn.GRUCell(27, latent_dim) # 26字母 + <S>
        self.vocal_out = nn.Linear(latent_dim, 27)
        self.listener_rnn = nn.GRU(27, latent_dim, batch_first=True)
        self.motor_cortex = nn.Linear(latent_dim, 2)

    def forward_speak(self, obs, lambda_ent=0.1):
        h = F.gelu(self.ln(self.vision_enc(obs)))
        mu = self.mu(h)
        log_var = torch.clamp(self.log_var(h), -5, 1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) * lambda_ent
        
        words_idxs, words_soft = [], []
        curr_h, curr_input = z, torch.zeros(1, 27)
        for _ in range(5): # 限制句长，强迫数学压缩
            curr_h = self.rnn_cell(curr_input, curr_h)
            logits = self.vocal_out(curr_h)
            soft_token = F.gumbel_softmax(logits, tau=0.8, hard=True)
            idx = torch.argmax(soft_token, dim=1).item()
            words_idxs.append(idx)
            words_soft.append(soft_token)
            curr_input = soft_token
            if idx == 26: break # <S>
        return words_idxs, torch.cat(words_soft, dim=0), 0.5 * torch.sum(1 + log_var + np.log(2 * np.pi))

    def forward_act(self, word_soft):
        _, final_h = self.listener_rnn(word_soft.unsqueeze(0))
        return torch.tanh(self.motor_cortex(final_h.squeeze(0)))

class Agent:
    def __init__(self, name):
        self.name, self.brain = name, POLOAR_Brain()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.003)
        self.reset()
    def reset(self):
        self.pos, self.energy, self.is_alive = torch.rand(2)*100, 500.0, True

def run_geometric_eden(generations=200):
    adam, eve = Agent("Adam"), Agent("Eve")
    alphabet = [chr(i) for i in range(97, 123)] + ["<S>"]

    for gen in range(generations):
        apple_pos = torch.rand(2) * 100
        adam.reset(); eve.reset()
        step, eaten = 0, 0
        log_book = []

        while adam.is_alive and eve.is_alive and step < 500:
            adam.optimizer.zero_grad(); eve.optimizer.zero_grad()

            # 1. 只有亚当能看到苹果
            obs_a = (apple_pos - adam.pos).detach() / 50.0
            idxs, soft, a_ent = adam.brain.forward_speak(obs_a.unsqueeze(0))
            word = "".join([alphabet[i] for i in idxs if i < 26])
            
            # 2. 夏娃全盲，只听亚当指挥
            e_move = eve.brain.forward_act(soft)
            
            # 3. 位移
            with torch.no_grad():
                eve.pos = torch.clamp(eve.pos + e_move.squeeze() * 8.0, 0, 100)

            # 4. POLOAR 结算：距离误差 + 生存压力
            dist = torch.norm(eve.pos - apple_pos)
            loss = dist + (500.0 / (min(adam.energy, eve.energy) + 1.0)) - 0.01 * a_ent
            loss.backward()
            adam.optimizer.step(); eve.optimizer.step()

            # 5. 能量清算
            cost = MAINT_BASE + len(word) * 0.1
            adam.energy -= cost; eve.energy -= cost

            if dist < 8.0:
                eaten += 1
                adam.energy += ENERGY_REWARD * 0.4
                eve.energy += ENERGY_REWARD * 0.6
                log_book.append(word)
                apple_pos = torch.rand(2) * 100
            
            if adam.energy <= 0 or eve.energy <= 0: adam.is_alive = False; eve.is_alive = False
            step += 1

        print(f"GEN {gen:3d} | Eaten: {eaten:2d} | E_Adam: {adam.energy:.1f} | Vocab: {set(log_book)}")

if __name__ == "__main__":
    run_geometric_eden()