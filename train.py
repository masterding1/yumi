import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import time
import json
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import math

# 创建保存图片的目录
os.makedirs('plots', exist_ok=True)

# 设置绘图风格
plt.style.use('seaborn-v0_8')  # 使用新版本的seaborn样式
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 类别映射
class_mapping = {
    "Bercak Daun": 0,
    "Daun Sehat": 1,
    "Hawar Daun": 2,
    "Karat Daun": 3
}
CLASS_NAMES = list(class_mapping.keys())

# 改进的早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.mode == 'min':
            score = -score  # 转换为最大化问题
        
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

# 改进的MobileNetV3
class ImprovedMobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(ImprovedMobileNetV3, self).__init__()
        # 加载预训练的MobileNetV3-Small
        self.mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # 获取最后一层的输入特征数
        last_channel = self.mobilenet.classifier[0].in_features
        
        # 添加通道注意力
        self.channel_attention = ChannelAttention(last_channel)
        
        # 添加空间注意力
        self.spatial_attention = SpatialAttention()
        
        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # 提取特征
        features = self.mobilenet.features(x)
        
        # 应用通道注意力
        channel_weights = self.channel_attention(features)
        features = features * channel_weights
        
        # 应用空间注意力
        spatial_weights = self.spatial_attention(features)
        features = features * spatial_weights
        
        # 全局平均池化
        features = nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        return output

# 增强的数据处理
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 改进的数据加载
def load_data(data_dir, batch_size=32, num_workers=4):
    # 加载完整数据集
    full_data = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # 计算类别权重
    class_counts = [0] * len(CLASS_NAMES)
    for _, label in full_data:
        class_counts[label] += 1
    
    # 使用平方根反比作为权重
    class_weights = [1.0 / math.sqrt(count) for count in class_counts]
    sample_weights = [class_weights[label] for _, label in full_data]
    
    # 划分数据集
    train_size = int(0.7 * len(full_data))
    val_size = int(0.15 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    
    train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])
    val_data.dataset.transform = test_transform
    test_data.dataset.transform = test_transform
    
    # 为训练集创建加权采样器
    train_weights = [sample_weights[i] for i in train_data.indices]
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 评估函数
def evaluate(model, loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total

def plot_training_history(history, model_name):
    """绘制训练历史"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    ax1.set_title(f'{model_name} - Loss over epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.plot(epochs, history['test_acc'], 'g-', label='Test Acc')
    ax2.set_title(f'{model_name} - Accuracy over epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

def train_model():
    print(f"\n{'='*50}")
    print(f"Training Improved MobileNetV3")
    print(f"{'='*50}")
    
    # 加载数据
    DATA_DIR = "/kaggle/input/yumishuju/yumi"
    train_loader, val_loader, test_loader = load_data(DATA_DIR, batch_size=32, num_workers=4)
    
    # 创建模型
    model = ImprovedMobileNetV3(num_classes=len(class_mapping))
    model = model.to(device)
    
    # 打印模型参数量
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # 训练记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': [],
        'time_per_epoch': []
    }
    
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=7, mode='max')
    
    for epoch in range(1, 51):  # 增加训练轮数到50
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/50")
        
        # 训练
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100. * correct / total:.2f}%'})
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # 验证
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 测试
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 记录时间
        epoch_time = time.time() - epoch_start_time
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['time_per_epoch'].append(epoch_time)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, 'best_improved_mobilenetv3.pth')
        
        # 早停检查
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型进行最终测试
    checkpoint = torch.load('best_improved_mobilenetv3.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    
    # 保存结果
    results = {
        'model_name': 'improved_mobilenetv3',
        'num_parameters': num_params,
        'final_test_loss': final_test_loss,
        'final_test_acc': final_test_acc,
        'best_val_acc': best_val_acc,
        'history': history
    }
    
    # 绘制训练历史
    plot_training_history(history, 'improved_mobilenetv3')
    
    with open('improved_mobilenetv3_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    # 训练改进的MobileNetV3
    results = train_model()
    
    # 打印结果
    print("\n" + "="*50)
    print("Training Results")
    print("="*50)
    print(f"Number of parameters: {results['num_parameters']:,}")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Final test accuracy: {results['final_test_acc']:.4f}")
    print(f"Average time per epoch: {np.mean(results['history']['time_per_epoch']):.2f}s")

# 保存总结报告
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'models': {
        'improved_mobilenetv3': results
    }
}

with open('training_summary.json', 'w') as f:
    json.dump(summary, f, indent=4) 