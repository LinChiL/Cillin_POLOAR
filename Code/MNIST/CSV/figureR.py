import csv
import matplotlib.pyplot as plt
import os

def plot_from_csv():
    # 读取CSV文件
    # 用户选择要读取的文件
    choice = input("请选择要读取的文件 (输入 'n' 读取 MNIST.csv，输入 '8' 读取 MNIST_E8.csv，输入 '16' 读取 MNIST_E16.csv): ")
    if choice.lower() == 'n':
        csv_path = r'f:\Entropy_Intell\Code\MNIST\CSV\MNIST.csv'
    elif choice.lower() == '8':
        csv_path = r'f:\Entropy_Intell\Code\MNIST\CSV\MNIST_E8.csv'
    elif choice.lower() == '16':
        csv_path = r'f:\Entropy_Intell\Code\MNIST\CSV\MNIST_E16.csv'
    else:
        print("无效输入，使用默认文件 MNIST.csv")
        csv_path = r'f:\Entropy_Intell\Code\MNIST\CSV\MNIST.csv'
    
    if not os.path.exists(csv_path):
        print("CSV文件不存在！")
        return
    
    # 读取数据（没有表头）
    lambdas = []
    L_values = []
    accuracies = []
    robustness = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lambdas.append(float(row[0]))  # Lambda
            L_values.append(float(row[1]))  # L
            accuracies.append(float(row[2]))  # Accuracy
            robustness.append(float(row[3]))  # Robustness
    
    # 创建保存目录
    save_dir = r'f:\Entropy_Intell\Code\MNIST\CSV'
    
    # 图表1：Lambda vs L（点图）- 使用对数刻度
    plt.figure(figsize=(10, 5))
    plt.scatter(lambdas, L_values, color='green', s=50, alpha=0.7)
    plt.xscale('log')
    plt.xlabel("Lambda (Entropy Driving Force)")
    plt.ylabel("Leverage L (Complexity / Budget)")
    plt.title("Lambda-L Relationship (POLOAR Phase Diagram)")
    plt.axhline(y=1.0, color='red', linestyle='--', label='Life Boundary (L=1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Lambda_L_Scatter.png'), dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {os.path.join(save_dir, 'Lambda_L_Scatter.png')}")
    
    # 图表2：L vs Performance（点图）- 使用线性刻度
    plt.figure(figsize=(10, 5))
    plt.scatter(L_values, accuracies, color='blue', s=50, alpha=0.7, label='Accuracy')
    plt.scatter(L_values, robustness, color='red', s=50, alpha=0.7, label='Robustness')
    plt.xlabel("Leverage L (Complexity / Budget)")
    plt.ylabel("Performance")
    plt.title("Performance vs L (POLOAR Life Curve)")
    plt.axvline(x=1.0, color='green', linestyle='--', label='Life Point (L=1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Performance_L_Scatter.png'), dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {os.path.join(save_dir, 'Performance_L_Scatter.png')}")
    
    plt.show()

if __name__ == '__main__':
    plot_from_csv()
