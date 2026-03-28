import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict
import matplotlib.animation as animation



class POLOAR_ParticleUniverse:
    def __init__(self, n_x=100, n_y=100, total_energy=20000.0, xi=1.0):
        self.n_x = n_x  # x 方向点数
        self.n_y = n_y  # y 方向点数
        self.n = n_x * n_y  # 总点数
        self.xi = xi
        self.total_energy = total_energy
        
        # GPU 支持
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 奇点：能量集中在中心 100 个像素
        self.energies = torch.full((n_x, n_y), 1e-9, device=self.device)
        mid_x = n_x // 2
        mid_y = n_y // 2
        self.energies[mid_x-5:mid_x+5, mid_y-5:mid_y+5] = total_energy / 100.0  # 每个点 200
        self.energies *= (self.total_energy / self.energies.sum())
        self.grads = self.energies / self.xi
        
        # 历史记录
        self.history_grads = []
        self.history_peaks = []
        
        # 预存固定数据（全部在GPU上）
        # 坐标网格（用于计算距离）
        x = torch.arange(n_x, device=self.device)
        y = torch.arange(n_y, device=self.device)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # 预存峰值检测卷积核
        self.max_kernel = torch.ones(1, 1, 3, 3, device=self.device)
        
    def compute_pressure(self, grads):
        """逻辑压力：能量从高梯度区向低梯度区流动"""
        log_grads = torch.log(grads + 1e-9)
        # 二维梯度计算
        pressure_x, pressure_y = torch.gradient(log_grads)
        # 返回梯度的模长作为压力
        pressure = torch.sqrt(pressure_x**2 + pressure_y**2)
        return pressure
    
    def compute_diffusion(self, energies):
        """扩散项：能量自然扩散（二维）"""
        # 二维扩散：考虑上下左右四个方向
        up = torch.roll(energies, 1, dims=0)
        down = torch.roll(energies, -1, dims=0)
        left = torch.roll(energies, 1, dims=1)
        right = torch.roll(energies, -1, dims=1)
        return 0.1 * (up + down + left + right - 4*energies)
    
    def find_particles(self, grads, threshold=0.1):
        """找出梯度峰值 = 粒子（二维，旧方法）"""
        grads_np = grads.cpu().numpy()
        
        # 使用二维最大值检测
        from scipy.ndimage import maximum_filter
        
        # 创建最大值过滤器（3x3窗口）
        max_filtered = maximum_filter(grads_np, size=3)
        
        # 找到局部最大值（等于过滤器输出且大于阈值）
        local_max = (grads_np == max_filtered) & (grads_np > threshold)
        
        # 获取峰值坐标
        y_coords, x_coords = np.where(local_max)
        
        # 组合成坐标列表
        peaks = [(x, y) for x, y in zip(x_coords, y_coords)]
        
        return peaks, None
    
    def find_particles_gpu(self, grads, threshold=0.1):
        """GPU 峰值检测，无 CPU 传输"""
        # 3x3 局部最大值检测
        max_pool = torch.nn.functional.max_pool2d(
            grads.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze()
        
        is_local_max = (grads == max_pool) & (grads > threshold)
        y_coords, x_coords = torch.where(is_local_max)
        
        # 直接返回坐标，不转 CPU
        return torch.stack([x_coords, y_coords], dim=1) if len(x_coords) > 0 else torch.empty(0, 2, device=self.device)
    
    def find_atoms(self):
        """查找稳定配对的原子（简化版，基于当前粒子）"""
        peaks = self.find_particles_gpu(self.grads, threshold=0.05)
        n_particles = peaks.shape[0]
        
        if n_particles < 2:
            return []
        
        atoms = []
        checked_pairs = set()
        
        # 计算所有粒子对的距离
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # 计算二维欧几里得距离
                dx = peaks[i, 0] - peaks[j, 0]
                dy = peaks[i, 1] - peaks[j, 1]
                dist = torch.sqrt(dx ** 2 + dy ** 2)
                
                # 如果距离小于5像素，认为是原子
                if dist < 5:
                    pair = tuple(sorted([i, j]))
                    if pair not in checked_pairs:
                        atoms.append((i, j, dist.item()))
                        checked_pairs.add(pair)
        
        return atoms
    
    def find_atoms_detailed(self):
        """查找稳定配对的原子，并分析类型"""
        peaks = self.find_particles_gpu(self.grads, threshold=0.05)
        n_particles = peaks.shape[0]
        
        if n_particles < 2:
            return []
        
        atoms = []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dx = peaks[i, 0] - peaks[j, 0]
                dy = peaks[i, 1] - peaks[j, 1]
                dist = torch.sqrt(dx ** 2 + dy ** 2)
                
                if dist < 8:  # 放宽到 8 像素
                    # 获取两个粒子的梯度强度（质量）
                    grad_i = self.grads[peaks[i, 1].long(), peaks[i, 0].long()].item()
                    grad_j = self.grads[peaks[j, 1].long(), peaks[j, 0].long()].item()
                    
                    # 计算原子类型
                    atom_type = self.classify_atom(dist.item(), grad_i, grad_j)
                    
                    atoms.append({
                        'pid1': i,
                        'pid2': j,
                        'distance': dist.item(),
                        'mass1': grad_i,
                        'mass2': grad_j,
                        'type': atom_type
                    })
        
        return atoms
    
    def classify_atom(self, distance, mass1, mass2):
        """根据距离和质量比分类原子"""
        mass_ratio = max(mass1, mass2) / (min(mass1, mass2) + 1e-9)
        
        if distance < 1.5:
            return "fusion"  # 几乎重合，可能正在融合
        elif distance < 2.5:
            if mass_ratio > 3:
                return "ionic"  # 离子键（一大一小）
            else:
                return "covalent"  # 共价键（大小相近）
        elif distance < 4.0:
            return "polar"  # 极性键
        elif distance < 6.0:
            return "hydrogen"  # 氢键
        else:
            return "van_der_Waals"  # 范德华力
    

    

    

    

    
    def analyze_particle_type(self, grads, pos):
        """分析粒子类型（基于形状，二维）"""
        window = 10
        x, y = pos
        
        # 获取二维窗口
        x_left = max(0, x - window)
        x_right = min(self.n_x, x + window)
        y_top = max(0, y - window)
        y_bottom = min(self.n_y, y + window)
        
        profile = grads[y_top:y_bottom, x_left:x_right].cpu().numpy()
        
        # 计算形状特征
        height = profile.max()
        
        # 找周围小峰（卫星峰）
        from scipy.ndimage import maximum_filter
        max_filtered = maximum_filter(profile, size=3)
        local_max = (profile == max_filtered) & (profile > height*0.3)
        peaks = np.sum(local_max)
        n_satellites = peaks - 1  # 减去主峰
        
        # 对称性（简化处理）
        symmetry = 0.0
        if profile.shape[0] > 1 and profile.shape[1] > 1:
            # 左右对称
            center_x = profile.shape[1] // 2
            left_half = profile[:, :center_x]
            right_half = profile[:, center_x+1:][:, ::-1]
            if left_half.size > 0 and right_half.size > 0:
                min_w = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_w]
                right_half = right_half[:, :min_w]
                if left_half.std() > 1e-10 and right_half.std() > 1e-10:
                    symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0,1]
        
        if n_satellites == 0:
            return "lepton"  # 单峰，像电子
        elif n_satellites == 2:
            return "meson"   # 三峰，像介子
        elif symmetry > 0.8:
            return "boson"   # 对称分布，像光子
        else:
            return "baryon"  # 其他，像重子
    
    def evolve(self, steps=50000, record_every=1000):
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
            
            # 3. 量子涨落（虚粒子对产生）- GPU版
            if np.random.rand() < 0.02:
                x = torch.randint(0, self.n_x, (1,), device=self.device)
                y = torch.randint(0, self.n_y, (1,), device=self.device)
                self.energies[y, x] += torch.randn(1, device=self.device) * 5.0
            
            # 4. 更新能量分布
            self.energies += pressure * 1.0 + diffusion
            self.energies.clamp_(min=1e-8)
            
            # 5. 能量守恒
            self.energies *= (self.total_energy / self.energies.sum())
            self.grads = self.energies / self.xi
            
            # 6. 粒子探测 - GPU版
            peaks = self.find_particles_gpu(self.grads, threshold=0.05)
            n_particles = peaks.shape[0]
            particle_counts.append(n_particles)
            
            # 7. 记录历史 - 只在需要时转CPU
            if step % record_every == 0:
                # 记录梯度历史
                self.history_grads.append(self.grads.clone())
                
                # 转换为CPU格式用于可视化
                peaks_cpu = peaks.cpu().numpy()
                self.history_peaks.append(peaks_cpu)
                
                # 分析粒子类型（需要CPU转换）
                type_counts = defaultdict(int)
                if n_particles > 0:
                    for i in range(n_particles):
                        x, y = peaks[i, 0].item(), peaks[i, 1].item()
                        ptype = self.analyze_particle_type(self.grads, (x, y))
                        type_counts[ptype] += 1
                        particle_types[ptype] += 1
                
                # 详细原子分析
                atoms = self.find_atoms_detailed()
                
                print(f"Step {step:6d} | Particles: {n_particles:3d} | "
                      f"Max Grad: {self.grads.max():.2f} | "
                      f"Types: {dict(type_counts)} | "
                      f"Atoms: {len(atoms)}")
                
                # 统计原子类型
                atom_types = defaultdict(int)
                for atom in atoms:
                    atom_types[atom['type']] += 1
                if atom_types:
                    print(f"  Atom types: {dict(atom_types)}")
                
                # 打印前几个原子的详细信息
                for atom in atoms[:3]:
                    print(f"    {atom['type']}: d={atom['distance']:.2f}, m1={atom['mass1']:.1f}, m2={atom['mass2']:.1f}")
        
        return particle_counts, particle_types
    
    def visualize(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 最终梯度分布（热力图）
        ax = axes[0, 0]
        im = ax.imshow(self.grads.cpu().numpy(), cmap='hot', aspect='equal')
        peaks = self.find_particles_gpu(self.grads, threshold=0.05)
        if peaks.shape[0] > 0:
            peaks_cpu = peaks.cpu().numpy()
            x_coords = peaks_cpu[:, 0]
            y_coords = peaks_cpu[:, 1]
            ax.scatter(x_coords, y_coords, color='blue', s=20, zorder=5)
        ax.set_title(f"Final Gradient Distribution ({peaks.shape[0]} particles)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="∇Ω")
        
        # 2. 能量分布（热力图）
        ax = axes[0, 1]
        im = ax.imshow(self.energies.cpu().numpy(), cmap='viridis', aspect='equal')
        ax.set_title("Energy Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="mc²")
        
        # 3. 粒子数量演化
        ax = axes[1, 0]
        if hasattr(self, 'particle_counts'):
            ax.plot(self.particle_counts, 'purple', linewidth=1)
            ax.set_title("Particle Count Evolution")
            ax.set_xlabel("Step")
            ax.set_ylabel("Number of Particles")
        else:
            ax.text(0.5, 0.5, "No particle count data", ha='center', va='center', transform=ax.transAxes)
        
        # 4. 粒子位置（当前时刻）
        ax = axes[1, 1]
        if peaks.shape[0] > 0:
            ax.scatter(x_coords, y_coords, color='red', s=30, alpha=0.7)
            ax.set_title("Current Particle Positions")
        else:
            ax.text(0.5, 0.5, "No particles detected", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_particle_motion(self):
        """可视化粒子运动（简化版）"""
        plt.figure(figsize=(10, 8))
        
        # 使用最新的粒子位置
        peaks = self.find_particles_gpu(self.grads, threshold=0.05)
        if peaks.shape[0] > 0:
            peaks_cpu = peaks.cpu().numpy()
            x_coords = peaks_cpu[:, 0]
            y_coords = peaks_cpu[:, 1]
            
            plt.scatter(x_coords, y_coords, color='red', s=30, alpha=0.7)
            plt.title("Current Particle Positions")
        else:
            plt.text(0.5, 0.5, "No particles detected", ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
    
    def visualize_with_bonds(self):
        """可视化粒子和原子键"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 梯度热图
        im = ax.imshow(self.grads.cpu().numpy(), cmap='hot', aspect='equal')
        plt.colorbar(im, ax=ax, label="∇Ω")
        
        # 粒子位置
        peaks = self.find_particles_gpu(self.grads, threshold=0.05)
        if peaks.shape[0] > 0:
            peaks_cpu = peaks.cpu().numpy()
            x_coords = peaks_cpu[:, 0]
            y_coords = peaks_cpu[:, 1]
            ax.scatter(x_coords, y_coords, color='blue', s=30, zorder=5)
        
        # 画原子键
        atoms = self.find_atoms_detailed()
        colors = {'covalent': 'cyan', 'ionic': 'orange', 'hydrogen': 'green',
                  'van_der_Waals': 'gray', 'polar': 'yellow', 'fusion': 'red'}
        
        for atom in atoms:
            # 获取粒子位置
            peaks_cpu = peaks.cpu().numpy()
            x1, y1 = peaks_cpu[atom['pid1']]
            x2, y2 = peaks_cpu[atom['pid2']]
            
            color = colors.get(atom['type'], 'white')
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
        
        ax.set_title(f"Particles and Bonds ({len(atoms)} atoms)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()
    
    def animate_evolution(self, interval=50):
        """创建2D演化动画"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 初始图像
        im = ax.imshow(self.history_grads[0].cpu().numpy(), cmap='hot', aspect='equal')
        plt.colorbar(im, ax=ax, label="∇Ω")
        ax.set_title("POLOAR Particle Universe Evolution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        def update(frame):
            # 更新梯度分布
            grad_data = self.history_grads[frame].cpu().numpy()
            im.set_data(grad_data)
            
            # 更新标题
            step = frame * 1000  # 假设记录间隔为1000
            ax.set_title(f"POLOAR Particle Universe Evolution - Step {step}")
            
            return [im]
        
        # 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history_grads),
            interval=interval, blit=True, repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return ani

# --- 运行模拟 ---
print("=" * 60)
print("POLOAR Particle Universe Simulation")
print("=" * 60)

universe = POLOAR_ParticleUniverse(n_x=100, n_y=100, total_energy=20000.0)
particle_counts, particle_types = universe.evolve(steps=50000, record_every=1000)
universe.particle_counts = particle_counts

print("\n" + "=" * 60)
print("FINAL STATISTICS")
print("=" * 60)
print(f"Total particles detected: {sum(particle_counts[-100:])/100:.1f} (avg last 100 steps)")
print(f"Particle type distribution:")
for ptype, count in particle_types.items():
    print(f"  {ptype}: {count}")

# 检测最终原子
final_atoms = universe.find_atoms()
print(f"\nFinal atoms detected: {len(final_atoms)}")
if final_atoms:
    print("Stable atomic pairs:")
    for pid1, pid2, dist in final_atoms[:5]:  # 只打印前5个
        print(f"  Atom {pid1}-{pid2}: distance={dist:.2f}")



universe.visualize()
universe.visualize_particle_motion()
universe.visualize_with_bonds()

# 创建演化动画
print("\nCreating evolution animation...")
universe.animate_evolution()