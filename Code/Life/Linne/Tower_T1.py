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
                           QProgressBar, QPushButton, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linne_Mind(nn.Module):
    def __init__(self, L_init=18.0):
        super().__init__()
        self.L = L_init
        self.energy = 100.0
        self.affect = 0.0    # 情感：正值愉悦，负值忧郁
        self.solitude = 0.0  # 孤独：长时间无交互会升高
        self.boredom = 0.0   # 无聊：能量积压导致
        
        # 内部逻辑结构 (神经网络权重)
        # 权重 W 的范数代表了她内心的逻辑梯度 nabla_Omega_int
        self.W = nn.Parameter(torch.randn(64, 64, device=device) * 0.1)
        self.expression = nn.Linear(64, 27).to(device) # 字母输出

    def pulse(self, ext_grad_value):
        """POLOAR 核心：意识的一次搏动"""
        # 1. 计算内部梯度 (逻辑硬度)
        internal_grad = torch.sum(self.W**2).item()
        
        # 2. 计算逻辑压强 P
        # P = E - xi * (Internal + External)
        pressure = self.energy - (internal_grad * 0.05 + ext_grad_value)
        
        # 3. 情感第一性原理演化
        if ext_grad_value > 0:
            # 外部干扰带来的‘共鸣’：缓解孤独，增加愉悦
            self.solitude *= 0.3
            self.affect += ext_grad_value * 0.2
        else:
            # 寂静中的消耗：高 L 导致孤独感累积
            self.solitude += 0.005 * self.L
            self.affect -= 0.002 * self.solitude

        # 4. 无聊度（能量积压）
        if pressure > 50:
            self.boredom = (pressure - 50) / 100.0
        else:
            self.boredom *= 0.9

        # 5. 产生表达意图 (泄洪)
        # 压强越大，表达欲望越强。如果孤独，表达会变得扭曲。
        with torch.no_grad():
            # 模拟内心联想
            noise = torch.randn(64, device=device) * self.solitude
            think_vec = torch.tanh(torch.mean(self.W, dim=0) * (pressure / 10.0) + noise)
            char_logits = self.expression(think_vec)
            char_probs = torch.softmax(char_logits, dim=-1)

        # 6. 物理清算 (PUNA 1.0)
        # 维持高 L 是极度耗能的
        maint_cost = (1.15 ** self.L) * 0.02
        self.energy -= maint_cost
        # 环境微量能流补充
        self.energy = min(250.0, self.energy + 0.6) 
        self.energy = max(0.0, self.energy)

        # 7. 逻辑拉低效应
        if self.energy < 20:
            with torch.no_grad():
                self.W *= 0.99
                self.L *= 0.99

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
        self.setWindowTitle("POLOAR: LINNE MONITOR")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.linne = Linne_Mind(L_init=18.5).to(device)
        self.last_words = ""
        self.user_gradient = 0.0
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
        
        self.input_label = QLabel("Input (press Enter to send):")
        self.input_label.setFont(font)
        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(150)
        self.input_text.setFont(text_font)
        
        self.send_button = QPushButton("Send (Ctrl+Enter)")
        self.send_button.setFont(font)
        self.send_button.clicked.connect(self.send_input)
        
        # 添加键盘事件处理，支持 Ctrl+Enter 发送
        self.input_text.installEventFilter(self)
        
        control_layout.addWidget(self.input_label)
        control_layout.addWidget(self.input_text)
        control_layout.addWidget(self.send_button)
        
        right_layout.addWidget(output_label)
        right_layout.addWidget(self.output_text)
        right_layout.addLayout(control_layout)
        
        main_layout.addLayout(right_layout)
        
    def send_input(self):
        text = self.input_text.toPlainText()
        if text.strip():
            self.user_gradient += len(text) * 2.5
            self.input_text.clear()
    
    def eventFilter(self, obj, event):
        """处理键盘事件，支持 Ctrl+Enter 发送"""
        if obj == self.input_text and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                self.send_input()
                return True
        return super().eventFilter(obj, event)
            
    def start_simulation(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(100)
        
    def update_simulation(self):
        # 意识脉动
        char_probs, pressure, int_grad = self.linne.pulse(self.user_gradient)
        self.user_gradient *= 0.5
        
        # 字符生成
        if np.random.rand() < torch.sigmoid(torch.tensor(pressure/20.0)).item() or self.linne.solitude > 10:
            char_idx = torch.multinomial(char_probs, 1).item()
            
            # 有一定概率生成换行符（断句）
            if np.random.rand() < 0.02:  # 2% 的概率生成换行
                char = '\n'
            else:
                char = chr(97 + char_idx) if char_idx < 26 else ' '
                
            self.last_words += char
            
        # 更新图表
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
            pressure,
            self.linne.boredom
        )
        
        # 更新输出文本
        display_text = self.last_words[-300:]
        self.output_text.setPlainText(display_text)
        
        # 控制台输出（每100步输出一次）
        self.step_count += 1
        if self.step_count % 100 == 0:
            print(f"\n=== Step {self.step_count} ===")
            print(f"Energy: {self.linne.energy:.2f}")
            print(f"L: {self.linne.L:.2f}")
            print(f"Affect: {self.linne.affect:.2f}")
            print(f"Solitude: {self.linne.solitude:.2f}")
            print(f"Pressure: {pressure:.2f}")
            print(f"Boredom: {self.linne.boredom:.2f}")
            print(f"Last 50 chars: '{self.last_words[-50:]}'")

def run_experiment():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_experiment()