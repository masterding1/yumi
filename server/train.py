import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 类别映射
class_mapping = {
    "叶斑病": 0,
    "健康玉米": 1,
    "叶锈病": 2,
    "叶黄病": 3
}
CLASS_NAMES = list(class_mapping.keys())

# 创建保存目录
os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

class CustomShuffleNet(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomShuffleNet, self).__init__()
        # 加载预训练的ShuffleNetV2
        self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
        
        # 移除原始的分类层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 添加自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 数据加载
DATA_DIR = "./data/yumi"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"请将数据集放在 {os.path.abspath(DATA_DIR)} 目录下，每个类别一个子文件夹")
    print("类别文件夹名称应为：")
    for class_name in CLASS_NAMES:
        print(f"- {class_name}")
    exit(0)

# 加载数据集
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)  # 70% 训练集
val_size = int(0.15 * total_size)   # 15% 验证集
test_size = total_size - train_size - val_size  # 15% 测试集

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# 修改验证集和测试集的transform
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

def plot_metrics(train_losses, val_losses, train_accs, val_accs, epoch):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/metrics_epoch_{epoch}.png')
    plt.close()

def plot_confusion_matrix(true_labels, predictions, epoch):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_epoch_{epoch}.png')
    plt.close()

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 添加L2正则化
        l2_lambda = 0.01
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels

def main():
    # 初始化模型
    model = CustomShuffleNet(num_classes=len(CLASS_NAMES))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=3, 
                                                   verbose=True)
    
    # 训练记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    # 训练循环
    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        
        # 验证阶段
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 绘制指标图
        plot_metrics(train_losses, val_losses, train_accs, val_accs, epoch)
        
        # 每5个epoch绘制一次混淆矩阵
        if epoch % 5 == 0:
            plot_confusion_matrix(val_labels, val_preds, epoch)
        
        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model/model.pth')
            print(f"  New best model saved with val accuracy: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # 在测试集上评估最终模型
    model.load_state_dict(torch.load('model/model.pth'))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    
    # 绘制最终的混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, 'final')

if __name__ == "__main__":
    main()
