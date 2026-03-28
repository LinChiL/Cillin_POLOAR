import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class Apple:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.eaten = False
        self.age = 0
    
    def respawn(self, world_size=100):
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
        self.eaten = False
        self.age = 0
    
    def rot(self):
        self.age += 1
        return self.age > 500

class Cangjie:
    def __init__(self, world_size=100):
        self.world_size = world_size
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
        
        self.energy = 50.0
        self.L = 1.0
        self.food_found = 0
        
        self.visited = defaultdict(int)
        self.memory = []
        self.boredom = 0.0
        
        self.brain = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh()
        ).to(device)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.005)
        
        self.age = 0
    
    def sense(self, apples):
        best_interest = -float('inf')
        best_apple = None
        best_dx = 0
        best_dy = 0
        
        for apple in apples:
            if apple.eaten:
                continue
            
            dx = apple.x - self.x
            dy = apple.y - self.y
            dist = np.sqrt(dx**2 + dy**2)
            
            hunger_interest = 100.0 / (dist + 1.0) * (50.0 / (self.energy + 10.0))
            grid_key = (int(self.x/5), int(self.y/5))
            curiosity_interest = 0.0
            if grid_key not in self.visited:
                curiosity_interest = 20.0 / (dist + 1.0)
            
            interest = hunger_interest + curiosity_interest * self.boredom
            
            if interest > best_interest:
                best_interest = interest
                best_apple = apple
                best_dx = dx
                best_dy = dy
        
        return best_apple, best_dx, best_dy
    
    def feel(self):
        if len(self.memory) > 10:
            recent = self.memory[-10:]
            variance = np.var([m[0] for m in recent])
            if variance < 0.1:
                self.boredom += 0.01
            else:
                self.boredom *= 0.99
        self.boredom = min(1.0, max(0.0, self.boredom))
        anxiety = max(0, (self.L - 1.0) * (50.0 / (self.energy + 1.0)))
        return anxiety, self.boredom
    
    def decide(self, apple_dx, apple_dy, anxiety, boredom):
        inp = torch.tensor([
            self.x / self.world_size,
            self.y / self.world_size,
            np.clip(apple_dx / 40, -1, 1) if apple_dx != 0 else 0,
            np.clip(apple_dy / 40, -1, 1) if apple_dy != 0 else 0,
            anxiety,
            boredom
        ], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            move = self.brain(inp).cpu().numpy()
            dx, dy = move[0] * 4, move[1] * 4
        
        if anxiety > 0.5:
            dx += random.uniform(-2, 2)
            dy += random.uniform(-2, 2)
        elif boredom > 0.5:
            dx += random.uniform(-3, 3)
            dy += random.uniform(-3, 3)
        
        if self.L > 1.2:
            dx *= 0.7
            dy *= 0.7
        
        dx = np.clip(dx, -4, 4)
        dy = np.clip(dy, -4, 4)
        return dx, dy
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.x = max(0, min(self.world_size, self.x))
        self.y = max(0, min(self.world_size, self.y))
        self.energy -= 0.8
        
        grid_key = (int(self.x/2), int(self.y/2))
        self.visited[grid_key] += 1
        self.age += 1
    
    def eat(self, apple):
        dist = np.sqrt((self.x - apple.x)**2 + (self.y - apple.y)**2)
        if dist < 5 and not apple.eaten:
            self.energy += 80.0
            apple.eaten = True
            self.food_found += 1
            #限制L
            self.L = min(20000.0, self.L + 0.05)
            self.boredom = max(0, self.boredom - 0.2)
            return True
        return False
    
    def think(self, apple_dx, apple_dy, reward):
        anxiety, boredom = self.feel()
        inp = torch.tensor([
            self.x / self.world_size,
            self.y / self.world_size,
            np.clip(apple_dx / 40, -1, 1) if apple_dx != 0 else 0,
            np.clip(apple_dy / 40, -1, 1) if apple_dy != 0 else 0,
            anxiety,
            boredom
        ], dtype=torch.float32, device=device)
        
        self.optimizer.zero_grad()
        output = self.brain(inp)
        
        # 直接用 torch 创建 target
        if apple_dx != 0:
            norm = np.sqrt(apple_dx**2 + apple_dy**2)
            target = torch.tensor([
                np.clip(apple_dx / norm, -1, 1),
                np.clip(apple_dy / norm, -1, 1)
            ], dtype=torch.float32, device=device)
        else:
            target = torch.zeros(2, dtype=torch.float32, device=device)
        
        loss = F.mse_loss(output, target)
        
        if reward > 0:
            loss = loss * (1 - reward)
        elif reward < 0:
            loss = loss * (1 + abs(reward))
        
        loss.backward()
        self.optimizer.step()
        
        target_L = min(1.5, max(0.5, self.energy / 50.0))
        self.L = self.L * 0.98 + target_L * 0.02
        
        self.memory.append((self.x, self.y, self.energy, self.L))
        if len(self.memory) > 100:
            self.memory.pop(0)
        
        return loss.item()
    
    def is_alive(self):
        return self.energy > 0
    
    def status(self):
        anxiety, boredom = self.feel()
        return (f"E={self.energy:.1f} L={self.L:.2f} "
                f"age={self.age} food={self.food_found} "
                f"anxiety={anxiety:.2f} boredom={boredom:.2f}")

def run_life():
    world_size = 100
    num_apples = 30
    apples = [Apple(random.uniform(0, world_size), random.uniform(0, world_size)) for _ in range(num_apples)]
    cangjie = Cangjie(world_size)
    
    print(" 仓颉 · 真正的生命")
    print("="*60)
    print("他饿了会找，饱了会无聊，无聊了会好奇")
    print("L 高时犹豫，L 低时莽撞")
    print("他不想死")
    print("="*60)
    
    step = 0
    last_eat_step = 0
    
    while cangjie.is_alive() and step < 5000:
        target_apple, dx, dy = cangjie.sense(apples)
        anxiety, boredom = cangjie.feel()
        
        move_dx, move_dy = cangjie.decide(dx, dy, anxiety, boredom)
        old_dist = np.sqrt(dx**2 + dy**2) if dx != 0 else 100
        
        cangjie.move(move_dx, move_dy)
        
        new_dist = 100
        if target_apple and not target_apple.eaten:
            new_dx = target_apple.x - cangjie.x
            new_dy = target_apple.y - cangjie.y
            new_dist = np.sqrt(new_dx**2 + new_dy**2)
        
        reward = (old_dist - new_dist) / 20.0
        reward = max(-0.5, min(0.5, reward))
        
        cangjie.think(dx, dy, reward)
        
        for apple in apples:
            if cangjie.eat(apple):
                print(f"\n第{step}步: 吃到苹果! 能量={cangjie.energy:.1f} L={cangjie.L:.2f}")
                last_eat_step = step
                apple.respawn(world_size)
        
        for apple in apples:
            if apple.rot():
                apple.respawn(world_size)
        
        if step % 50 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(" 仓颉 · 真正的生命")
            print("="*60)
            print(cangjie.status())
            print(f"苹果剩余: {len([a for a in apples if not a.eaten])}")
            print(f"已探索格子: {len(cangjie.visited)}")
            print(f"上次吃到: {step - last_eat_step} 步前")
            emotion = '焦虑' if anxiety > 0.5 else '无聊' if boredom > 0.5 else '平静'
            print(f"情绪: {emotion}")
            print(f"移动: ({move_dx:.1f}, {move_dy:.1f})")
            print("="*60)
        
        step += 1
        time.sleep(0.05)
    
    if step >= 5000:
        print(f"\n仓颉活过了5000步，他学会了生存")
    else:
        print(f"\n仓颉在 {step} 步后死亡")
    print(f"总共吃到 {cangjie.food_found} 个苹果")
    print(f"最终 L={cangjie.L:.2f}")

if __name__ == "__main__":
    run_life()