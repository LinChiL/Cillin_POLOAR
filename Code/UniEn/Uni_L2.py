import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict
import matplotlib.animation as animation

class Particle:
    def __init__(self, pid, pos, grad, step):
        self.pid = pid
        self.pos = pos          # 逻辑坐标
        self.grad = grad        # 梯度强度（质量）
        self.birth = step
        self.history_pos = [pos]
        self.history_grad = [grad]

class POLOAR_ParticleUniverse:
    def __init__(self, total_points=1000, total_energy=20000.0, xi=1.0):
        self.n = total_points
        self.xi = xi
        self.total_energy = total_energy
        
        # 奇点：能量集中在中心 20 个像素
        self.energies = torch.full((total_points,), 1e-9)
        mid = total_points // 2
        self.energies[mid-10:mid+10] = total_energy / 20.0  # 每个点 1000
        self.energies *= (self.total_energy / self.energies.sum())
        self.grads = self.energies / self.xi
        
        # 历史记录
        self.history_grads = []
        self.history_peaks = []
        self.particle_tracks = defaultdict(list)  # 粒子轨迹
        self.particle_lifetimes = defaultdict(int)
        self.next_particle_id = 0
        
        # 粒子运动追踪
        self.particles = {}  # 存储粒子对象
        self.prev_positions = {}
        self.prev_dist = {}
        self.prev_ratio = {}
        
    def compute_pressure(self, grads):
        """逻辑压力：能量从高梯度区向低梯度区流动"""
        log_grads = torch.log(grads + 1e-9)
        pressure = -torch.gradient(log_grads)[0]
        return pressure
    
    def compute_diffusion(self, energies):
        """扩散项：能量自然扩散"""
        return 0.1 * (torch.roll(energies, 1) + torch.roll(energies, -1) - 2*energies)
    
    def find_particles(self, grads, threshold=0.1):
        """找出梯度峰值 = 粒子"""
        grads_np = grads.numpy()
        peaks, props = find_peaks(grads_np, height=threshold, prominence=0.05)
        return peaks, props
    
    def track_particles(self, current_peaks, step):
        """追踪粒子轨迹，计算寿命"""
        new_tracks = {}
        
        # 匹配现有粒子
        matched = set()
        for pid, (last_pos, birth_step) in list(self.particle_tracks.items()):
            # 找最近的新峰
            if len(current_peaks) == 0:
                continue
            distances = [abs(p - last_pos) for p in current_peaks]
            min_dist = min(distances)
            if min_dist < 5:  # 移动不超过5个像素
                idx = distances.index(min_dist)
                new_pos = current_peaks[idx]
                new_tracks[pid] = (new_pos, birth_step)
                matched.add(idx)
                self.particle_lifetimes[pid] = step - birth_step
        
        # 新粒子
        for i, pos in enumerate(current_peaks):
            if i not in matched:
                pid = self.next_particle_id
                self.next_particle_id += 1
                new_tracks[pid] = (pos, step)
                self.particle_lifetimes[pid] = 0
        
        self.particle_tracks = new_tracks
        return len(new_tracks)
    
    def track_particles_detailed(self, current_peaks, current_grads, step):
        """详细追踪粒子运动"""
        new_tracks = {}
        current_positions = list(current_peaks)
        
        # 1. 匹配现有粒子
        matched = set()
        for pid, particle in self.particles.items():
            if len(current_positions) == 0:
                continue
            # 找最近的新位置
            distances = [abs(p - particle.pos) for p in current_positions]
            min_dist = min(distances)
            if min_dist < 10:  # 移动不超过10像素
                idx = distances.index(min_dist)
                new_pos = current_positions[idx]
                particle.pos = new_pos
                particle.history_pos.append(new_pos)
                particle.grad = current_grads[new_pos].item()
                particle.history_grad.append(particle.grad)
                new_tracks[pid] = particle
                matched.add(idx)
        
        # 2. 新粒子
        for i, pos in enumerate(current_positions):
            if i not in matched:
                pid = self.next_particle_id
                self.next_particle_id += 1
                new_particle = Particle(pid, pos, current_grads[pos].item(), step)
                new_tracks[pid] = new_particle
        
        self.particles = new_tracks
        return len(self.particles)
    
    def analyze_interactions(self, step):
        """分析粒子之间的互动"""
        positions = [p.pos for p in self.particles.values()]
        grads = [p.grad for p in self.particles.values()]
        pids = list(self.particles.keys())
        
        if len(positions) < 2:
            return []
        
        interactions = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = abs(positions[i] - positions[j])
                
                # 距离很近 → 可能耦合
                if dist < 15:
                    # 计算能量相关性
                    grad_i = grads[i]
                    grad_j = grads[j]
                    grad_ratio = grad_i / (grad_j + 1e-9)
                    
                    # 检查是否在绕转（速度方向）
                    if hasattr(self, 'prev_positions'):
                        prev_i = self.prev_positions.get(pids[i], positions[i])
                        prev_j = self.prev_positions.get(pids[j], positions[j])
                        v_i = positions[i] - prev_i
                        v_j = positions[j] - prev_j
                        
                        # 相对速度方向
                        rel_v = v_i - v_j
                        
                        # 如果距离稳定且相对速度垂直 → 绕转
                        if abs(dist - self.prev_dist.get((i,j), dist)) < 2:
                            if abs(rel_v) > 0.1:
                                interactions.append({
                                    'type': 'orbiting',
                                    'particles': (pids[i], pids[j]),
                                    'distance': dist,
                                    'step': step
                                })
                    
                    # 能量交换
                    if abs(grad_ratio - self.prev_ratio.get((i,j), grad_ratio)) > 0.5:
                        interactions.append({
                            'type': 'energy_exchange',
                            'particles': (pids[i], pids[j]),
                            'ratio': grad_ratio,
                            'step': step
                        })
        
        # 记录历史
        self.prev_positions = {pid: p.pos for pid, p in self.particles.items()}
        self.prev_dist = {(i,j): abs(positions[i] - positions[j])
                          for i in range(len(positions))
                          for j in range(i+1, len(positions))}
        self.prev_ratio = {(i,j): grads[i]/(grads[j]+1e-9)
                          for i in range(len(positions))
                          for j in range(i+1, len(positions))}
        
        return interactions
    
    def analyze_particle_type(self, grads, pos):
        """分析粒子类型（基于形状）"""
        window = 20
        left = max(0, pos - window)
        right = min(self.n, pos + window)
        profile = grads[left:right].numpy()
        
        # 计算形状特征
        height = profile.max()
        width = np.sum(profile > height * 0.5)
        
        # 找周围小峰（卫星峰）
        sub_peaks, _ = find_peaks(profile, height=height*0.3)
        n_satellites = len(sub_peaks) - 1  # 减去主峰
        
        # 对称性
        center = len(profile) // 2
        left_profile = profile[:center]
        right_profile = profile[center+1:][::-1]
        
        # 确保两个数组长度相同
        min_len = min(len(left_profile), len(right_profile))
        left_profile = left_profile[:min_len]
        right_profile = right_profile[:min_len]
        
        # 检查是否有足够的数据点
        if len(left_profile) < 2 or len(right_profile) < 2:
            symmetry = 0.0
        else:
            # 检查标准差是否为零
            left_std = np.std(left_profile)
            right_std = np.std(right_profile)
            if left_std < 1e-10 or right_std < 1e-10:
                symmetry = 0.0
            else:
                symmetry = np.corrcoef(left_profile, right_profile)[0,1]
        
        if n_satellites == 0:
            return "lepton"  # 单峰，像电子
        elif n_satellites == 2:
            return "meson"   # 三峰，像介子
        elif symmetry > 0.8:
            return "boson"   # 对称分布，像光子
        else:
            return "baryon"  # 其他，像重子
    
    def evolve(self, steps=50000, record_every=500):
        print("Starting POLOAR Particle Universe...")
        print(f"Total Energy: {self.total_energy}, Points: {self.n}")
        
        particle_counts = []
        particle_types = defaultdict(int)
        
        for step in range(steps):
            # 1. PUNA 1.0：能量决定梯度
            self.grads = self.energies / self.xi
            
            # 2. 动力学：压力驱动 + 扩散
            pressure = self.compute_pressure(self.grads)
            diffusion = self.compute_diffusion(self.energies)
            
            # 3. 量子涨落（虚粒子对产生）
            if np.random.rand() < 0.02:
                pos = np.random.randint(0, self.n)
                self.energies[pos] += np.random.randn() * 5.0
            
            # 4. 更新能量分布
            self.energies += pressure * 1.0 + diffusion
            self.energies.clamp_(min=1e-8)
            
            # 5. 能量守恒
            self.energies *= (self.total_energy / self.energies.sum())
            self.grads = self.energies / self.xi
            
            # 6. 粒子探测
            peaks, props = self.find_particles(self.grads, threshold=0.05)
            
            # 7. 追踪运动
            n_particles = self.track_particles_detailed(peaks, self.grads, step)
            particle_counts.append(n_particles)
            
            # 8. 分析互动
            interactions = self.analyze_interactions(step)
            if interactions and step % 100 == 0:
                print(f"Step {step}: {len(interactions)} interactions detected")
                for inter in interactions[:3]:  # 只打印前3个
                    print(f"  {inter['type']}: particles {inter['particles']}")
            
            # 9. 记录历史
            if step % record_every == 0:
                self.history_grads.append(self.grads.clone())
                self.history_peaks.append(peaks)
                
                # 分析粒子类型
                type_counts = defaultdict(int)
                for pos in peaks:
                    ptype = self.analyze_particle_type(self.grads, pos)
                    type_counts[ptype] += 1
                    particle_types[ptype] += 1
                
                print(f"Step {step:6d} | Particles: {n_particles:3d} | "
                      f"Max Grad: {self.grads.max():.2f} | "
                      f"Types: {dict(type_counts)}")
            
            # 10. 终止条件：达到稳态
            # if step > 5000 and np.std(particle_counts[-500:]) < 2:
            #     print(f"Steady state reached at step {step}")
            #     break
        
        return particle_counts, particle_types
    
    def visualize(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 最终梯度分布
        ax = axes[0, 0]
        ax.plot(self.grads.numpy(), 'b-', linewidth=1)
        peaks, _ = self.find_particles(self.grads)
        ax.scatter(peaks, self.grads[peaks].numpy(), color='red', s=30, zorder=5)
        ax.set_title(f"Final Gradient Distribution ({len(peaks)} particles)")
        ax.set_xlabel("Logical Coordinate")
        ax.set_ylabel("∇Ω")
        
        # 2. 粒子寿命分布
        ax = axes[0, 1]
        lifetimes = list(self.particle_lifetimes.values())
        if lifetimes:
            ax.hist(lifetimes, bins=50, color='green', alpha=0.7)
            ax.set_title("Particle Lifetimes")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Count")
        
        # 3. 粒子数量演化
        ax = axes[1, 0]
        if hasattr(self, 'particle_count_history'):
            ax.plot(self.particle_count_history, 'purple', linewidth=1)
            ax.set_title("Particle Count Evolution")
            ax.set_xlabel("Step")
            ax.set_ylabel("Number of Particles")
        
        # 4. 梯度分布演化热图
        ax = axes[1, 1]
        if self.history_grads:
            grad_history = torch.stack(self.history_grads).numpy()
            im = ax.imshow(grad_history.T, aspect='auto', cmap='hot', 
                          extent=[0, len(self.history_grads), 0, self.n])
            ax.set_title("Gradient Evolution (Space-Time)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Logical Coordinate")
            plt.colorbar(im, ax=ax, label="∇Ω")
        
        # 5. 粒子轨迹
        ax = axes[0, 2]
        for pid, (pos, birth) in list(self.particle_tracks.items()):
            # 简化：只显示最终位置
            ax.scatter(pos, pid, s=50, alpha=0.7)
        ax.set_title("Particle Positions")
        ax.set_xlabel("Logical Coordinate")
        ax.set_ylabel("Particle ID")
        
        # 6. 能量分布
        ax = axes[1, 2]
        ax.plot(self.energies.numpy(), 'orange', linewidth=1)
        ax.set_title("Energy Distribution")
        ax.set_xlabel("Logical Coordinate")
        ax.set_ylabel("mc²")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_particle_motion(self):
        """可视化粒子运动轨迹"""
        plt.figure(figsize=(12, 6))
        
        for pid, particle in self.particles.items():
            positions = particle.history_pos
            grads = particle.history_grad
            
            # 轨迹线
            plt.plot(positions, color='blue', alpha=0.5, linewidth=1)
            # 起点
            plt.scatter(positions[0], 0, color='green', s=50, marker='o')
            # 终点
            plt.scatter(positions[-1], 0, color='red', s=50, marker='s')
            # 质量大小（颜色深浅）
            for i, (pos, grad) in enumerate(zip(positions, grads)):
                plt.scatter(pos, 0, color='red', alpha=grad/100, s=10)
        
        plt.title("Particle Trajectories (Time → Y direction?)")
        plt.xlabel("Logical Coordinate")
        plt.ylabel("Time (not to scale)")
        plt.show()

# --- 运行模拟 ---
print("=" * 60)
print("POLOAR Particle Universe Simulation")
print("=" * 60)

universe = POLOAR_ParticleUniverse(total_points=1000, total_energy=20000.0)
particle_counts, particle_types = universe.evolve(steps=15000, record_every=500)

print("\n" + "=" * 60)
print("FINAL STATISTICS")
print("=" * 60)
print(f"Total particles detected: {sum(particle_counts[-100:])/100:.1f} (avg last 100 steps)")
print(f"Particle type distribution:")
for ptype, count in particle_types.items():
    print(f"  {ptype}: {count}")

universe.visualize()
universe.visualize_particle_motion()