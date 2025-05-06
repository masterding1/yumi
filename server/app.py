from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime
import numpy as np
from PIL import Image
import io
import base64
import time
import os
import torch
from scipy.special import expit

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
    img_array = np.array(image)  # 保留原始数据类型
    # 转换为RGB（如果是RGBA）
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # 归一化
    img_array = img_array / 255.0
    # 标准化
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    img_array = (img_array - mean) / std
    # 转换为NCHW格式
    img_array = img_array.transpose(2, 0, 1)
    img_array = img_array.reshape(1, 3, target_size[0], target_size[1])
    
    # 转换为张量并移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_data = torch.tensor(img_array).to(device)
    
    # 检查数据类型并转换为float32
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)
    
    # 将张量转换为NumPy数组
    numpy_data = tensor_data.cpu().numpy()
    
    return numpy_data

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
    # 获取其他参数
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    time_info = request.form.get('time', current_time) if request.form else request.json.get('time', current_time) if request.is_json else current_time
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
        raw_probabilities = output[0][0]
        probabilities = expit(raw_probabilities)  # 应用Sigmoid函数
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])
        disease = CLASS_NAMES[predicted_class_idx]
        inference_time = time.time() - start_time
        
        # 生成建议
        suggestion = f"检测到{disease}，{disease}详情信息：\n"
        
        if disease == "叶斑病":
            suggestion += "症状：初期在叶片上出现水渍状褪绿小点，后扩展为圆形或椭圆形病斑，边缘红褐色，中央灰白色。\n"
            suggestion += "病因：由玉米大斑病菌引起，属于真菌性病害。\n"
            suggestion += "危害：影响光合作用，导致植株早衰，籽粒不饱满。\n"
            suggestion += "防治建议：选用抗病品种，实行轮作，及时清除病残体，发病初期可喷洒代森锰锌、百菌清等杀菌剂。"
        elif disease == "叶锈病":
            suggestion += "症状：叶片上出现橙黄色或铁锈色粉状孢子堆，后期变为黑褐色。\n"
            suggestion += "病因：由玉米锈菌引起，高温高湿有利于病害发生。\n"
            suggestion += "危害：破坏叶片组织，影响光合作用，导致叶片早枯。\n"
            suggestion += "防治建议：选用抗病品种，避免种植过密，保持田间通风透光，发病初期喷洒三唑酮等杀菌剂。"
        elif disease == "叶黄病":
            suggestion += "症状：叶片从边缘或叶尖开始发黄，叶脉仍保持绿色。\n"
            suggestion += "病因：可能由缺素、病毒侵染或生理性因素引起。\n"
            suggestion += "危害：影响植株正常生长发育，光合效率降低。\n"
            suggestion += "防治建议：根据土壤情况补充相应元素肥料，防治传毒昆虫，加强田间管理。"
        else:  # 健康玉米
            suggestion += "症状：叶片浓绿，生长健壮，无明显病斑或异常。\n"
            suggestion += "防治建议：保持良好的田间管理，适时适量施肥浇水。"
        
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

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "running",
        "message": "玉米叶片病害识别服务器正在运行",
        "endpoints": {
            "/predict": "POST - 上传图片进行病害识别"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
