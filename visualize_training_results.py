import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# 设置绘图风格
plt.style.use('seaborn')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# 创建保存图表的目录
os.makedirs('training_plots', exist_ok=True)

# 模型数据
models = {
    'shufflenet_v2_x1_0': {
        'parameters': 1257704,
        'test_acc': 0.9650,
        'val_acc': 0.9783,
        'avg_time': 237.14,
        'epochs': 14,
        'history': {
            'train_loss': [0.4369, 0.1319, 0.0715, 0.0628, 0.0712, 0.0543, 0.0288, 0.0252, 0.0424, 0.0264, 0.0268, 0.0207, 0.0078, 0.0059],
            'train_acc': [0.9039, 0.9636, 0.9782, 0.9807, 0.9829, 0.9843, 0.9907, 0.9925, 0.9871, 0.9925, 0.9929, 0.9921, 0.9979, 0.9989],
            'val_loss': [0.1125, 0.0993, 0.0767, 0.0977, 0.0819, 0.0784, 0.0728, 0.1407, 0.0796, 0.0826, 0.1358, 0.1107, 0.1104, 0.0955],
            'val_acc': [0.9650, 0.9750, 0.9767, 0.9633, 0.9783, 0.9767, 0.9733, 0.9667, 0.9750, 0.9767, 0.9700, 0.9750, 0.9733, 0.9750]
        }
    },
    'mobilenet_v3_small': {
        'parameters': 929316,
        'test_acc': 0.9767,
        'val_acc': 0.9817,
        'avg_time': 237.85,
        'epochs': 20,
        'history': {
            'train_loss': [0.2470, 0.0814, 0.0759, 0.0637, 0.0280, 0.0334, 0.0623, 0.0301, 0.0106, 0.0098, 0.0039, 0.0060, 0.0033, 0.0023, 0.0024, 0.0018, 0.0011, 0.0016, 0.0007, 0.0005],
            'train_acc': [0.9136, 0.9704, 0.9754, 0.9779, 0.9918, 0.9889, 0.9821, 0.9918, 0.9971, 0.9961, 0.9993, 0.9982, 0.9989, 0.9996, 0.9993, 0.9993, 0.9996, 0.9996, 1.0000, 1.0000],
            'val_loss': [0.7333, 0.3401, 0.1398, 0.1495, 0.1975, 0.1354, 0.1318, 0.0775, 0.0972, 0.1258, 0.1024, 0.1059, 0.0773, 0.0746, 0.0799, 0.0810, 0.0839, 0.0800, 0.0803, 0.0805],
            'val_acc': [0.8267, 0.9050, 0.9500, 0.9617, 0.9600, 0.9717, 0.9550, 0.9750, 0.9783, 0.9667, 0.9767, 0.9733, 0.9783, 0.9800, 0.9817, 0.9817, 0.9817, 0.9817, 0.9817, 0.9817]
        }
    },
    'efficientnet_b0': {
        'parameters': 4012672,
        'test_acc': 0.9783,
        'val_acc': 0.9850,
        'avg_time': 239.53,
        'epochs': 11,
        'history': {
            'train_loss': [0.2721, 0.1485, 0.1012, 0.0862, 0.1068, 0.0523, 0.0500, 0.0502, 0.0170, 0.0130, 0.0112],
            'train_acc': [0.9168, 0.9589, 0.9671, 0.9732, 0.9661, 0.9825, 0.9854, 0.9861, 0.9939, 0.9971, 0.9961],
            'val_loss': [0.1697, 0.0912, 0.1080, 0.0723, 0.1400, 0.0786, 0.1007, 0.0985, 0.0905, 0.0833, 0.0827],
            'val_acc': [0.9517, 0.9700, 0.9700, 0.9750, 0.9483, 0.9750, 0.9717, 0.9783, 0.9833, 0.9833, 0.9850]
        }
    }
}

def plot_parameters_comparison():
    """绘制模型参数量比较"""
    plt.figure(figsize=(10, 6))
    model_names = list(models.keys())
    parameters = [models[model]['parameters'] for model in model_names]
    
    bars = plt.bar(model_names, parameters, color='skyblue')
    plt.title('Model Parameters Comparison', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Parameters', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_plots/parameters_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_comparison():
    """绘制准确率比较"""
    plt.figure(figsize=(10, 6))
    model_names = list(models.keys())
    test_acc = [models[model]['test_acc'] for model in model_names]
    val_acc = [models[model]['val_acc'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, test_acc, width, label='Test Accuracy', color='lightgreen')
    plt.bar(x + width/2, val_acc, width, label='Validation Accuracy', color='lightblue')
    
    plt.title('Model Accuracy Comparison', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(test_acc):
        plt.text(i - width/2, v, f'{v:.4f}', ha='center', va='bottom')
    for i, v in enumerate(val_acc):
        plt.text(i + width/2, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_plots/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_time_comparison():
    """绘制训练时间比较"""
    plt.figure(figsize=(10, 6))
    model_names = list(models.keys())
    times = [models[model]['avg_time'] for model in model_names]
    
    bars = plt.bar(model_names, times, color='lightcoral')
    plt.title('Average Training Time per Epoch', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_plots/training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves():
    """绘制学习曲线"""
    plt.figure(figsize=(15, 10))
    
    for model_name, model_data in models.items():
        history = model_data['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 训练损失
        plt.plot(epochs, history['train_loss'], 
                label=f'{model_name} (Train Loss)',
                linestyle='-', linewidth=2)
        
        # 验证损失
        plt.plot(epochs, history['val_loss'],
                label=f'{model_name} (Val Loss)',
                linestyle='--', linewidth=2)
    
    plt.title('Learning Curves - Loss', fontsize=14, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_plots/learning_curves_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 准确率学习曲线
    plt.figure(figsize=(15, 10))
    
    for model_name, model_data in models.items():
        history = model_data['history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        # 训练准确率
        plt.plot(epochs, history['train_acc'],
                label=f'{model_name} (Train Acc)',
                linestyle='-', linewidth=2)
        
        # 验证准确率
        plt.plot(epochs, history['val_acc'],
                label=f'{model_name} (Val Acc)',
                linestyle='--', linewidth=2)
    
    plt.title('Learning Curves - Accuracy', fontsize=14, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_plots/learning_curves_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_speed():
    """绘制收敛速度比较"""
    plt.figure(figsize=(10, 6))
    model_names = list(models.keys())
    epochs = [models[model]['epochs'] for model in model_names]
    
    bars = plt.bar(model_names, epochs, color='lightgreen')
    plt.title('Convergence Speed Comparison', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Epochs to Converge', fontsize=12)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_plots/convergence_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_iteration_process():
    """绘制每个模型的迭代过程"""
    plt.figure(figsize=(15, 10))
    
    for model_name, model_data in models.items():
        history = model_data['history']
        iterations = range(1, len(history['train_loss']) + 1)
        
        # 训练损失
        plt.plot(iterations, history['train_loss'], 
                label=f'{model_name} (Train Loss)',
                linestyle='-', linewidth=2,
                marker='o', markersize=4)
        
        # 验证损失
        plt.plot(iterations, history['val_loss'],
                label=f'{model_name} (Val Loss)',
                linestyle='--', linewidth=2,
                marker='s', markersize=4)
    
    plt.title('Model Iteration Process', fontsize=14, pad=20)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度为整数
    plt.xticks(range(1, max(len(models[model]['history']['train_loss']) for model in models) + 1))
    
    plt.tight_layout()
    plt.savefig('training_plots/iteration_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 准确率迭代过程
    plt.figure(figsize=(15, 10))
    
    for model_name, model_data in models.items():
        history = model_data['history']
        iterations = range(1, len(history['train_acc']) + 1)
        
        # 训练准确率
        plt.plot(iterations, history['train_acc'],
                label=f'{model_name} (Train Acc)',
                linestyle='-', linewidth=2,
                marker='o', markersize=4)
        
        # 验证准确率
        plt.plot(iterations, history['val_acc'],
                label=f'{model_name} (Val Acc)',
                linestyle='--', linewidth=2,
                marker='s', markersize=4)
    
    plt.title('Model Iteration Process - Accuracy', fontsize=14, pad=20)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度为整数
    plt.xticks(range(1, max(len(models[model]['history']['train_acc']) for model in models) + 1))
    
    plt.tight_layout()
    plt.savefig('training_plots/iteration_process_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 生成所有可视化图表
    plot_parameters_comparison()
    plot_accuracy_comparison()
    plot_training_time_comparison()
    plot_learning_curves()
    plot_convergence_speed()
    plot_iteration_process()
    
    print("所有可视化图表已保存到 'training_plots' 目录：")
    print("1. parameters_comparison.png - 模型参数量比较")
    print("2. accuracy_comparison.png - 准确率比较")
    print("3. training_time_comparison.png - 训练时间比较")
    print("4. learning_curves_loss.png - 损失学习曲线")
    print("5. learning_curves_accuracy.png - 准确率学习曲线")
    print("6. convergence_speed_comparison.png - 收敛速度比较")
    print("7. iteration_process.png - 模型迭代过程（损失）")
    print("8. iteration_process_accuracy.png - 模型迭代过程（准确率）")

if __name__ == "__main__":
    main() 