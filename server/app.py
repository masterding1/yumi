from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
import onnxruntime as ort
import time

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 加载模型
MODEL_PATH = 'model/mobilenetv3.onnx'
print(f" 模型加载成功: {MODEL_PATH} ")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type
print(f" 模型输入名称: {input_name}, 形状: {input_shape}, 类型: {input_type} ")

# 类别映射
CLASS_NAMES = ["叶斑病", "健康玉米", "叶锈病", "叶黄病"]  # 根据您的实际类别修改

@app.route('/')
def index():
    return jsonify({"message": "玉米病害识别API服务正在运行"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({"success": False, "error": "未找到图像数据"})
    
    try:
        # 解码base64图像
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # 打印图像信息用于调试
        print(f"图像大小: {image.size}, 模式: {image.mode}")
        
        # 预处理图像
        start_time = time.time()
        
        # 调整大小并确保是RGB模式
        image = image.resize((224, 224))
        image = image.convert('RGB')
        
        # 转换为numpy数组并确保是float32类型
        image_array = np.array(image, dtype=np.float32)
        
        # 归一化
        image_array = image_array / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        image_array = (image_array - mean) / std
        
        # 调整维度顺序 (H,W,C) -> (C,H,W)
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # 添加批次维度
        image_array = np.expand_dims(image_array, axis=0)
        
        # 确保类型是float32
        image_array = image_array.astype(np.float32)
        
        # 打印数组信息用于调试
        print(f"输入数组形状: {image_array.shape}, 类型: {image_array.dtype}")
        print(f"输入数组范围: [{np.min(image_array)}, {np.max(image_array)}]")
        
        # 进行预测
        outputs = session.run(None, {input_name: image_array})
        probabilities = outputs[0][0]
        
        # 获取预测结果
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # 确保置信度在0-1之间
        confidence = max(0.0, min(1.0, confidence))
        
        # 打印预测结果用于调试
        print(f"预测类别: {predicted_class}, 置信度: {confidence}")
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        # 构建响应
        class_probs = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_probs[class_name] = float(probabilities[i])
        
        return jsonify({
            "success": True,
            "disease": predicted_class,
            "confidence": confidence,  # 这个值在0-1之间
            "inference_time": inference_time,
            "class_probabilities": class_probs
        })
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)