import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 创建保存图片的目录
os.makedirs('plots', exist_ok=True)

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def load_results():
    """加载所有模型的训练结果"""
    all_results = {}
    model_names = ["shufflenet_v2_x1_0", "mobilenet_v3_small", "efficientnet_b0"]
    
    for model_name in model_names:
        try:
            with open(f'{model_name}_results.json', 'r') as f:
                all_results[model_name] = json.load(f)
            print(f"Loaded results for {model_name}")
        except FileNotFoundError:
            print(f"Warning: Results file not found for {model_name}")
    
    return all_results

def plot_training_history(history, model_name):
    """绘制训练历史"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制损失
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'g-', label='Test Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Loss over epochs', fontsize=14, pad=20)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # 绘制准确率
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'g-', label='Test Acc', linewidth=2)
    ax2.set_title(f'{model_name} - Accuracy over epochs', fontsize=14, pad=20)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join('plots', f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")

def plot_model_comparison(all_results):
    """绘制模型比较结果"""
    models = list(all_results.keys())
    metrics = ['num_parameters', 'final_test_acc', 'best_val_acc']
    metric_names = ['Parameters', 'Test Accuracy', 'Validation Accuracy']
    
    # 创建三个独立的图表，而不是一个大的子图
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.figure(figsize=(10, 6))
        values = [all_results[model][metric] for model in models]
        
        if metric == 'num_parameters':
            # 使用对数刻度显示参数量
            bars = plt.bar(models, np.log10(values), color='skyblue')
            plt.ylabel('Log10(Parameters)', fontsize=12)
        else:
            bars = plt.bar(models, values, color='lightgreen')
            plt.ylabel(name, fontsize=12)
        
        plt.title(name, fontsize=14, pad=20)
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            if metric == 'num_parameters':
                label = f'{10**height:,.0f}'
            else:
                label = f'{height:.4f}'
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join('plots', f'model_comparison_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {name} comparison plot to {save_path}")

def plot_learning_curves(all_results):
    """绘制学习曲线比较"""
    plt.figure(figsize=(15, 10))
    
    for model_name, results in all_results.items():
        history = results['history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        plt.plot(epochs, history['train_acc'], 
                label=f'{model_name} (Train)', 
                linestyle='-', 
                linewidth=2)
        plt.plot(epochs, history['val_acc'], 
                label=f'{model_name} (Val)', 
                linestyle='--', 
                linewidth=2)
    
    plt.title('Learning Curves Comparison', fontsize=14, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    save_path = os.path.join('plots', 'learning_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves plot to {save_path}")

def plot_training_time_comparison(all_results):
    """绘制训练时间比较"""
    plt.figure(figsize=(12, 6))
    
    models = list(all_results.keys())
    avg_times = [np.mean(results['history']['time_per_epoch']) for results in all_results.values()]
    
    bars = plt.bar(models, avg_times, color='lightblue')
    plt.title('Average Training Time per Epoch', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join('plots', 'training_time_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training time comparison plot to {save_path}")

def main():
    # 加载结果
    all_results = load_results()
    
    if not all_results:
        print("No results found. Please run the training script first.")
        return
    
    # 为每个模型绘制训练历史
    for model_name, results in all_results.items():
        plot_training_history(results['history'], model_name)
    
    # 绘制模型比较图
    plot_model_comparison(all_results)
    
    # 绘制学习曲线比较
    plot_learning_curves(all_results)
    
    # 绘制训练时间比较
    plot_training_time_comparison(all_results)
    
    print("\nAll visualizations have been saved to the 'plots' directory.")

if __name__ == "__main__":
    main() 