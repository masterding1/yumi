from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime
import numpy as np
from PIL import Image
import io
import base64
import time
import os

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 加载ONNX模型
MODEL_PATH = os.path.join('model', 'mobilenetv3.onnx')
try:
    session = onnxruntime.InferenceSession(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    session = None

# 类别映射
CLASS_NAMES = ["叶斑病", "健康玉米", "叶锈病", "叶黄病"]

# 图像预处理
def preprocess_image(image, target_size=(224, 224)):
    # 调整大小
    image = image.resize(target_size)
    # 转换为numpy数组
    img_array = np.array(image).astype(np.float32)
    # 转换为RGB（如果是RGBA）
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # 归一化
    img_array = img_array / 255.0
    # 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    img_array = (img_array - mean) / std
    # 转换为NCHW格式
    img_array = img_array.transpose(2, 0, 1)
    img_array = img_array.reshape(1, 3, target_size[0], target_size[1])
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        # 检查是否为JSON请求中的base64图像
        if request.is_json:
            data = request.json
            image_base64 = data.get('image')
            if image_base64:
                try:
                    # 解码base64图像
                    image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    return jsonify({"success": False, "error": f"无法解析图像: {str(e)}"})
            else:
                return jsonify({"success": False, "error": "未找到图像数据"})
        else:
            return jsonify({"success": False, "error": "请求中未包含图像"})
    else:
        # 从表单数据中获取图像
        image_file = request.files['image']
        try:
            image = Image.open(image_file)
            # 保存图片到 uploads 文件夹
            uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            # 生成唯一文件名
            filename = f"{int(time.time())}_{np.random.randint(1000,9999)}.jpg"
            save_path = os.path.join(uploads_dir, filename)
            image.save(save_path)
        except Exception as e:
            return jsonify({"success": False, "error": f"无法打开图像: {str(e)}"})
    
    # 获取其他参数
    time_info = request.form.get('time', '未知') if request.form else request.json.get('time', '未知') if request.is_json else '未知'
    location = request.form.get('location', '未知') if request.form else request.json.get('location', '未知') if request.is_json else '未知'
    growth_stage = request.form.get('growth_stage', '未知') if request.form else request.json.get('growth_stage', '未知') if request.is_json else '未知'
    
    # 检查模型是否加载成功
    if session is None:
        return jsonify({
            "success": False,
            "error": "模型未加载，请检查服务器日志"
        })
    
    try:
        # 预处理图像
        start_time = time.time()
        input_data = preprocess_image(image)
        
        # 执行推理
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data})
        
        # 处理结果
        probabilities = output[0][0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])
        disease = CLASS_NAMES[predicted_class_idx]
        inference_time = time.time() - start_time
        
        # 生成建议
        suggestion = f"检测到{disease}。建议：\n"
        if growth_stage == "苗期":
            if disease == "叶斑病":
                suggestion += "苗期叶斑病建议使用多菌灵等杀菌剂喷洒，保持田间通风。"
            elif disease == "叶锈病":
                suggestion += "苗期叶锈病建议清除病叶，使用三唑酮类药剂防治。"
            elif disease == "叶黄病":
                suggestion += "苗期叶黄病需检查土壤养分，适当追施氮肥。"
            else:
                suggestion += "健康玉米苗，继续保持良好的田间管理。"
        elif growth_stage == "拔节期":
            if disease == "叶斑病":
                suggestion += "拔节期叶斑病可喷洒代森锰锌等保护性杀菌剂。"
            elif disease == "叶锈病":
                suggestion += "拔节期叶锈病可喷洒三唑酮类药剂防治。"
            elif disease == "叶黄病":
                suggestion += "拔节期叶黄病需检查灌溉和养分状况，适当调整。"
            else:
                suggestion += "健康玉米，继续保持良好的田间管理。"
        else:
            suggestion += "建议根据实际生长期选择合适药剂，并注意田间管理。"
        
        suggestion += f"\n当前地点：{location}，时间：{time_info}"
        
        # 构建类别概率字典
        class_probs = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_probs[class_name] = float(probabilities[i])
        
        # 返回结果
        return jsonify({
            "success": True,
            "disease": disease,
            "confidence": confidence,
            "inference_time": inference_time,
            "class_probabilities": class_probs,
            "suggestion": suggestion
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"处理图像时出错: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
