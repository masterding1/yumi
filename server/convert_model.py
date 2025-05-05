import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 类别映射
CLASS_NAMES = ["Bercak Daun", "Daun Sehat", "Hawar Daun", "Karat Daun"]

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

# 改进的MobileNetV3模型
class ImprovedMobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(ImprovedMobileNetV3, self).__init__()
        # 加载预训练的MobileNetV3-Small
        self.mobilenet = models.mobilenet_v3_small(weights=None)
        
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

def load_model():
    print("加载模型架构...")
    model = ImprovedMobileNetV3(num_classes=len(CLASS_NAMES))
    
    print("加载模型权重...")
    try:
        # 加载模型权重
        checkpoint = torch.load('d:/yumixiaochengxu/best_improved_mobilenetv3.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("[成功] 模型权重加载成功!")
        
        # 打印模型结构
        print("\n模型结构:")
        print(model)
        
        # 验证模型是否正确加载
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print("\n测试推理输出形状:", output.shape)
            print("测试推理输出:", torch.nn.functional.softmax(output, dim=1))
        
        return model
    except Exception as e:
        print("[错误] 加载模型权重时出错:", str(e))
        raise

def convert_to_onnx(model):
    print("\n转换为ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 确保model目录存在
    import os
    os.makedirs('model', exist_ok=True)
    
    # 导出ONNX模型
    torch.onnx.export(model,
                     dummy_input,
                     'model/mobilenetv3.onnx',
                     export_params=True,
                     opset_version=11,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    print("[成功] 模型已导出为ONNX格式")

def verify_onnx():
    print("\n验证ONNX模型...")
    # 加载并检查ONNX模型
    onnx_model = onnx.load("model/mobilenetv3.onnx")
    onnx.checker.check_model(onnx_model)
    print("[成功] ONNX模型结构有效")
    
    # 测试ONNX模型推理
    print("\n测试ONNX推理...")
    ort_session = onnxruntime.InferenceSession("model/mobilenetv3.onnx")
    
    # 准备测试数据
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    
    # 运行推理
    ort_outputs = ort_session.run(None, ort_inputs)
    print("[成功] ONNX推理成功")
    print("输出形状:", ort_outputs[0].shape)
    
    # 打印预测概率
    probabilities = ort_outputs[0][0]
    print("\n预测概率:")
    for class_name, prob in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {prob:.4f}")

def main():
    try:
        # 1. 加载PyTorch模型
        model = load_model()
        
        # 2. 转换为ONNX
        convert_to_onnx(model)
        
        # 3. 验证ONNX模型
        verify_onnx()
        
        print("\n[成功] 模型转换完成!")
        
    except Exception as e:
        print("\n[错误] 模型转换过程中出错:", str(e))
        raise

if __name__ == "__main__":
    main()
