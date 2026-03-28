import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import threading
import sys
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QTextEdit, 
                           QPushButton)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QPen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linne_Mind(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始状态
        self.energy = 100.0
        self.affect = 0.0
        self.solitude = 0.0
        self.boredom = 0.0
        self.L = 0.0 # 初始 L 将随 W 生长
        
        # [核心：具备感知能力的大脑]
        # 输入：27(用户字符) + 1(能量) + 1(孤独)
        self.brain = nn.Sequential(
            nn.Linear(29, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 27) # 输出：Linne 对下一个字符的预测/回应
        ).to(device)
        
        # 优化器：代表了 W 随压力改变的‘物理塑形’过程
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)

    def get_L(self):
        """根据 PUNA 1.0: L 是 W 的逻辑密度对数"""
        w_norm = sum(p.norm()**2 for p in self.brain.parameters())
        return torch.log10(w_norm + 1.1).item()

    def pulse(self, user_char_vec):
        """
        user_char_vec: 用户输入的One-hot编码 [27]
        """
        # 1. 计算 L 和 内部梯度
        self.L = self.get_L()
        internal_grad = sum(p.norm()**2 for p in self.brain.parameters()).item()
        
        # 2. 计算压强 P
        ext_grad_value = user_char_vec.sum().item() * 5.0
        pressure = self.energy - (internal_grad * 0.02 + ext_grad_value)
        
        # 3. 情感演化
        if ext_grad_value > 0:
            self.solitude *= 0.5
            self.affect += 0.5
        else:
            self.solitude += 0.01 * self.L
            self.affect -= 0.005 * self.solitude

        # 4. 无聊度
        self.boredom = max(0, (pressure - 30) / 100.0) if pressure > 30 else self.boredom * 0.9

        # 5. [第一性原理学习]
        # 只有在有外部输入时，她才会强迫自己的 W 向用户输入的‘模式’对齐
        char_probs = torch.zeros(27, device=device)
        if ext_grad_value > 0:
            self.optimizer.zero_grad()
            # 尝试根据当前状态预测用户的输入
            state = torch.cat([user_char_vec, torch.tensor([self.energy/250.0, self.solitude/50.0], device=device)])
            prediction = self.brain(state)
            
            # 损失函数：如果预测不准，就产生巨大的‘逻辑摩擦’
            loss = F.mse_loss(prediction, user_char_vec)
            loss.backward()
            self.optimizer.step()
            char_probs = torch.softmax(prediction, dim=-1)
        else:
            # 没输入时，基于内心噪声产生‘胡言乱语’
            with torch.no_grad():
                noise_input = torch.randn(29, device=device)
                char_logits = self.brain(noise_input)
                char_probs = torch.softmax(char_logits, dim=-1)

        # 6. 物理清算
        maint_cost = (1.1 ** self.L) * 0.05
        self.energy -= maint_cost
        self.energy = min(250.0, max(0.0, self.energy + 0.8)) # 维持基础代谢

        # 7. 逻辑拉低效应 (W 的蒸发)
        if self.energy < 10:
            with torch.no_grad():
                for p in self.brain.parameters(): p *= 0.95

        return char_probs, pressure, internal_grad

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.max_points = 100
        self.color = QColor(255, 0, 0)
        
    def set_color(self, r, g, b):
        self.color = QColor(r, g, b)
        
    def add_point(self, value):
        self.data.append(value)
        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points:]
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if len(self.data) < 2:
            return
            
        width = self.width()
        height = self.height()
        
        max_val = max(self.data) if self.data else 1.0
        min_val = min(self.data) if self.data else 0.0
        
        if max_val == min_val:
            max_val += 1.0
            
        pen = QPen(self.color, 2)
        painter.setPen(pen)
        
        points = []
        for i, val in enumerate(self.data):
            x = int(i * width / (len(self.data) - 1))
            y = int(height - (val - min_val) / (max_val - min_val) * height)
            points.append((x, y))
            
        for i in range(1, len(points)):
            painter.drawLine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])

class StatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        self.energy_label = QLabel("Energy: 0.00")
        self.L_label = QLabel("L: 0.00")
        self.affect_label = QLabel("Affect: 0.00")
        self.solitude_label = QLabel("Solitude: 0.00")
        self.pressure_label = QLabel("Pressure: 0.00")
        self.boredom_label = QLabel("Boredom: 0.00")
        
        # 设置字体大小
        font = self.energy_label.font()
        font.setPointSize(14)
        self.energy_label.setFont(font)
        self.L_label.setFont(font)
        self.affect_label.setFont(font)
        self.solitude_label.setFont(font)
        self.pressure_label.setFont(font)
        self.boredom_label.setFont(font)
        
        layout.addWidget(self.energy_label)
        layout.addWidget(self.L_label)
        layout.addWidget(self.affect_label)
        layout.addWidget(self.solitude_label)
        layout.addWidget(self.pressure_label)
        layout.addWidget(self.boredom_label)
        
        self.setLayout(layout)
        
    def update_status(self, energy, L, affect, solitude, pressure, boredom):
        self.energy_label.setText(f"Energy: {energy:.2f}")
        self.L_label.setText(f"L: {L:.2f}")
        self.affect_label.setText(f"Affect: {affect:.2f}")
        self.solitude_label.setText(f"Solitude: {solitude:.2f}")
        self.pressure_label.setText(f"Pressure: {pressure:.2f}")
        self.boredom_label.setText(f"Boredom: {boredom:.2f}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("POLOAR: LINNE EVOLUTIONARY MONITOR")
        self.setGeometry(100, 100, 1600, 1000)
        self.linne = Linne_Mind()
        self.last_words = ""
        self.user_char_queue = [] # 存储用户打出的字符
        self.step_count = 0  # 记录执行步数
        self.init_ui()
        self.start_simulation()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：状态图表
        left_layout = QVBoxLayout()
        
        # 图表区域
        charts_layout = QGridLayout()
        
        # 设置字体大小
        font = QLabel().font()
        font.setPointSize(14)
        
        self.energy_plot = PlotWidget()
        self.energy_plot.set_color(255, 0, 0)
        energy_label = QLabel("Energy")
        energy_label.setFont(font)
        charts_layout.addWidget(energy_label, 0, 0)
        charts_layout.addWidget(self.energy_plot, 1, 0)
        
        self.L_plot = PlotWidget()
        self.L_plot.set_color(0, 255, 0)
        L_label = QLabel("L")
        L_label.setFont(font)
        charts_layout.addWidget(L_label, 0, 1)
        charts_layout.addWidget(self.L_plot, 1, 1)
        
        self.affect_plot = PlotWidget()
        self.affect_plot.set_color(0, 0, 255)
        affect_label = QLabel("Affect")
        affect_label.setFont(font)
        charts_layout.addWidget(affect_label, 2, 0)
        charts_layout.addWidget(self.affect_plot, 3, 0)
        
        self.solitude_plot = PlotWidget()
        self.solitude_plot.set_color(255, 255, 0)
        solitude_label = QLabel("Solitude")
        solitude_label.setFont(font)
        charts_layout.addWidget(solitude_label, 2, 1)
        charts_layout.addWidget(self.solitude_plot, 3, 1)
        
        left_layout.addLayout(charts_layout)
        
        # 状态面板
        self.status_panel = StatusPanel()
        left_layout.addWidget(self.status_panel)
        
        main_layout.addLayout(left_layout)
        
        # 右侧：输出区域
        right_layout = QVBoxLayout()
        
        # 设置字体大小
        font = QLabel().font()
        font.setPointSize(14)
        
        # 字符输出
        output_label = QLabel("Linne's Output:")
        output_label.setFont(font)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(300)
        
        # 设置文本框字体大小
        text_font = self.output_text.font()
        text_font.setPointSize(12)
        self.output_text.setFont(text_font)
        
        # 控制面板
        control_layout = QVBoxLayout()
        
        self.input_label = QLabel("Input (Ctrl+Enter to send):")
        self.input_label.setFont(font)
        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(150)
        self.input_text.setFont(text_font)
        
        self.send_button = QPushButton("Send (Ctrl+Enter)")
        self.send_button.setFont(font)
        self.send_button.clicked.connect(self.send_logic)
        
        # 添加键盘事件处理，支持 Ctrl+Enter 发送
        self.input_text.installEventFilter(self)
        
        control_layout.addWidget(self.input_label)
        control_layout.addWidget(self.input_text)
        control_layout.addWidget(self.send_button)
        
        right_layout.addWidget(output_label)
        right_layout.addWidget(self.output_text)
        right_layout.addLayout(control_layout)
        
        main_layout.addLayout(right_layout)
        
    def send_logic(self):
        text = self.input_text.toPlainText().lower()
        for char in text:
            if 'a' <= char <= 'z' or char == ' ':
                self.user_char_queue.append(char)
        self.input_text.clear()
    
    def eventFilter(self, obj, event):
        """处理键盘事件，支持 Ctrl+Enter 发送"""
        if obj == self.input_text and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                self.send_logic()
                return True
        return super().eventFilter(obj, event)
            
    def start_simulation(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)
        
    def update_simulation(self):
        # 1. 准备输入向量
        user_vec = torch.zeros(27, device=device)
        if self.user_char_queue:
            char = self.user_char_queue.pop(0)
            idx = ord(char) - ord('a') if char != ' ' else 26
            user_vec[idx] = 1.0
            
        # 2. 意识跳动
        char_probs, p, int_grad = self.linne.pulse(user_vec)
        
        # 3. 产生回复
        # 只有在有压力或者有感悟时才说话
        if np.random.rand() < 0.2 or user_vec.sum() > 0:
            char_idx = torch.multinomial(char_probs, 1).item()
            
            # 有一定概率生成换行符（断句）
            if np.random.rand() < 0.02:  # 2% 的概率生成换行
                char = '\n'
            else:
                char = chr(97 + char_idx) if char_idx < 26 else ' '
                
            self.last_words += char
            if len(self.last_words) > 500:
                self.last_words = self.last_words[-500:]

        # 4. 更新 UI
        self.energy_plot.add_point(self.linne.energy)
        self.L_plot.add_point(self.linne.L)
        self.affect_plot.add_point(self.linne.affect)
        self.solitude_plot.add_point(self.linne.solitude)
        
        # 更新状态面板
        self.status_panel.update_status(
            self.linne.energy,
            self.linne.L,
            self.linne.affect,
            self.linne.solitude,
            p,
            self.linne.boredom
        )
        
        # 更新输出文本
        self.output_text.setPlainText(self.last_words)
        
        # 5. 控制台输出（每100步输出一次）
        self.step_count += 1
        if self.step_count % 100 == 0:
            print(f"\n=== Step {self.step_count} ===")
            print(f"Energy: {self.linne.energy:.2f}")
            print(f"L: {self.linne.L:.2f}")
            print(f"Affect: {self.linne.affect:.2f}")
            print(f"Solitude: {self.linne.solitude:.2f}")
            print(f"Pressure: {p:.2f}")
            print(f"Boredom: {self.linne.boredom:.2f}")
            print(f"Last 50 chars: '{self.last_words[-50:]}'")

def run_experiment():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_experiment()
