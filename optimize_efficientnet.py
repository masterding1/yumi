import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.quantization import quantize_dynamic, get_default_qconfig
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import math  # 导入math模块用于NaN检查
import random
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold

# 设置绘图风格
plt.style.use('seaborn-v0_8')  # 使用更新的seaborn样式
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# 创建保存图表的目录
os.makedirs('optimization_plots', exist_ok=True)

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
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

# 简化的注意力机制
class SimplifiedChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SimplifiedChannelAttention, self).__init__()
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

class SimplifiedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SimplifiedSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

# 改进的DropBlock正则化
class ImprovedDropBlock2D(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.15):
        super(ImprovedDropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) < gamma).float()
        mask = self._compute_block_mask(mask)
        mask = mask.expand(-1, x.shape[1], -1, -1)
        
        # 添加残差连接
        out = x * mask * (mask.numel() / (mask.sum() + 1e-6))
        return out + x * (1 - mask)
    
    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
    
    def _compute_block_mask(self, mask):
        block_mask = nn.functional.max_pool2d(
            mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2
        )
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        return block_mask

# 改进的标签平滑
class ImprovedLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.15):
        super(ImprovedLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

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
def load_data(data_dir, batch_size=32, num_workers=4, k_folds=5):
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
    
    # 创建K折交叉验证数据加载器
    kfold_loaders = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 合并训练和验证数据用于交叉验证
    combined_data = torch.utils.data.ConcatDataset([train_data, val_data])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(combined_data)):
        train_subset = torch.utils.data.Subset(combined_data, train_idx)
        val_subset = torch.utils.data.Subset(combined_data, val_idx)
        
        # 为训练子集创建加权采样器
        train_subset_weights = [sample_weights[i] for i in train_idx]
        train_subset_sampler = WeightedRandomSampler(train_subset_weights, num_samples=len(train_subset_weights), replacement=True)
        
        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, sampler=train_subset_sampler, num_workers=num_workers)
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        kfold_loaders.append((train_loader_fold, val_loader_fold))
    
    return train_loader, val_loader, test_loader, kfold_loaders

# 简化的EfficientNet模型
class SimplifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimplifiedEfficientNet, self).__init__()
        # 加载预训练的EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 添加简化的通道注意力
        self.channel_attention = SimplifiedChannelAttention(1280)
        
        # 添加简化的空间注意力
        self.spatial_attention = SimplifiedSpatialAttention()
        
        # 替换原始分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
        # 添加DropBlock
        self.dropblock = ImprovedDropBlock2D(block_size=7, drop_prob=0.15)
        
    def forward(self, x):
        # 提取特征
        features = self.efficientnet.features(x)
        
        # 应用通道注意力
        channel_weights = self.channel_attention(features)
        features = features * channel_weights
        
        # 应用空间注意力
        spatial_weights = self.spatial_attention(features)
        features = features * spatial_weights
        
        # 应用DropBlock
        features = self.dropblock(features)
        
        # 全局平均池化
        features = nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        return output

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_structure(model):
    """分析模型结构"""
    structure = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            structure[name] = {
                'type': module.__class__.__name__,
                'in_channels': module.in_channels if hasattr(module, 'in_channels') else module.in_features,
                'out_channels': module.out_channels if hasattr(module, 'out_channels') else module.out_filters,
                'kernel_size': module.kernel_size if hasattr(module, 'kernel_size') else None,
                'parameters': sum(p.numel() for p in module.parameters())
            }
    return structure

def evaluate(model, test_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算每个类别的准确率
            for i in range(len(CLASS_NAMES)):
                mask = labels == i
                if mask.any():
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
    
    # 计算每个类别的准确率
    class_acc = [correct/total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    
    return total_loss / len(test_loader), correct / total, class_acc

def train_model(model, criterion, optimizer, scheduler, device, train_loader, val_loader, test_loader, num_epochs=50):
    """训练模型"""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    class_accs = {name: [] for name in CLASS_NAMES}
    best_val_acc = 0.0
    best_model_state = None
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
    
    # 使用混合精度训练
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 检查loss是否为NaN
            if math.isnan(loss.item()):
                print("Warning: NaN loss detected. Skipping batch.")
                continue
                
            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            
            # 添加梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}', 
                                 'acc': f'{100.*correct/total:.2f}%'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # 验证阶段
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        
        # 测试阶段
        test_loss, test_acc, class_acc = evaluate(model, test_loader, criterion, device)
        
        # 记录每个类别的准确率
        for i, acc in enumerate(class_acc):
            class_accs[CLASS_NAMES[i]].append(acc)
        
        # 记录损失和准确率
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印训练信息
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
        print('Class-wise accuracy:')
        for i, name in enumerate(CLASS_NAMES):
            print(f'{name}: {class_acc[i]:.4f}')
        
        # 学习率调度
        if not math.isnan(val_loss):  # 只在val_loss不是NaN时更新学习率
            scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        
        # 早停
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'class_accs': class_accs,
        'best_val_acc': best_val_acc
    }

def cross_validate(model_class, criterion_class, optimizer_class, scheduler_class, device, kfold_loaders, num_epochs=30):
    """交叉验证"""
    fold_results = []
    
    for fold, (train_loader, val_loader) in enumerate(kfold_loaders):
        print(f"\nFold {fold+1}/{len(kfold_loaders)}")
        
        # 创建模型
        model = model_class(num_classes=len(CLASS_NAMES))
        model = model.to(device)
        
        # 创建损失函数
        criterion = criterion_class()
        
        # 创建优化器
        optimizer = optimizer_class(model.parameters(), lr=0.0001, weight_decay=0.01)
        
        # 创建学习率调度器
        scheduler = scheduler_class(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # 训练模型
        results = train_model(model, criterion, optimizer, scheduler, device, 
                            train_loader, val_loader, None, num_epochs=num_epochs)
        
        fold_results.append(results)
        
        # 打印折叠结果
        print(f"Fold {fold+1} - Best Validation Accuracy: {results['best_val_acc']:.4f}")
    
    # 计算平均结果
    avg_best_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
    print(f"\nCross-validation average best validation accuracy: {avg_best_val_acc:.4f}")
    
    return fold_results, avg_best_val_acc

def plot_training_history(results):
    """绘制训练历史"""
    # 1. 损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.plot(results['test_losses'], label='Test Loss')
    plt.title('Training Loss History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimization_plots/loss_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 准确率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(results['train_accs'], label='Train Acc')
    plt.plot(results['val_accs'], label='Val Acc')
    plt.plot(results['test_accs'], label='Test Acc')
    plt.title('Training Accuracy History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimization_plots/accuracy_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 类别准确率曲线
    plt.figure(figsize=(12, 6))
    for name, accs in results['class_accs'].items():
        plt.plot(accs, label=name)
    plt.title('Class-wise Accuracy History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimization_plots/class_accuracy_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_accuracy(class_accs):
    """绘制类别准确率"""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    # 获取最后一个epoch的准确率
    final_accs = [accs[-1] for accs in class_accs.values()]
    
    plt.bar(x, final_accs, width, color='skyblue')
    plt.title('Class-wise Accuracy', fontsize=14)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x, CLASS_NAMES, rotation=45)
    
    # 添加数值标签
    for i, v in enumerate(final_accs):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('optimization_plots/class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载数据
    DATA_DIR = "/kaggle/input/yumishuju/yumi"
    train_loader, val_loader, test_loader, kfold_loaders = load_data(DATA_DIR, batch_size=32, num_workers=4, k_folds=5)
    
    # 交叉验证
    print("Starting cross-validation...")
    fold_results, avg_best_val_acc = cross_validate(
        SimplifiedEfficientNet, 
        ImprovedLabelSmoothingLoss, 
        optim.AdamW, 
        optim.lr_scheduler.ReduceLROnPlateau, 
        device, 
        kfold_loaders, 
        num_epochs=30
    )
    
    # 加载最终模型
    model = SimplifiedEfficientNet(num_classes=len(CLASS_NAMES))
    model = model.to(device)
    
    # 设置损失函数
    criterion = ImprovedLabelSmoothingLoss(smoothing=0.15)
    
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练最终模型
    print("\nTraining final model...")
    results = train_model(model, criterion, optimizer, scheduler, device, 
                         train_loader, val_loader, test_loader, num_epochs=50)
    
    # 绘制类别准确率
    plot_class_accuracy(results['class_accs'])
    
    # 绘制训练历史
    plot_training_history(results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'efficientnet_optimization_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n优化结果已保存到 {results_file}")
    
    # 打印训练结果
    print("\n训练结果摘要")
    print("="*50)
    print(f"交叉验证平均最佳验证准确率: {avg_best_val_acc:.4f}")
    print(f"最终模型最佳验证准确率: {results['best_val_acc']:.4f}")
    print(f"最终训练损失: {results['train_losses'][-1]:.4f}")
    print(f"最终验证损失: {results['val_losses'][-1]:.4f}")
    print(f"最终测试损失: {results['test_losses'][-1]:.4f}")
    
    print("\n最终类别准确率")
    print("="*50)
    print(f"{'类别':<15} {'准确率':<10}")
    print("-"*50)
    for name, accs in results['class_accs'].items():
        print(f"{name:<15} {accs[-1]:.4f}")
    
    # 计算并打印模型参数量
    total_params = count_parameters(model)
    print(f"\n模型总参数量: {total_params:,}")

if __name__ == "__main__":
    main() 