from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import UNet
import io
import matplotlib.pyplot as plt
import cv2
import base64
import time

app = Flask(__name__)
CORS(app)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, num_classes=4)
model.load_state_dict(torch.load('models1_2/best.pth', map_location=device))
model.to(device)
model.eval()

# 图像预处理
def preprocess_image(image_file):
    img = Image.open(image_file).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    return input_tensor, np.array(img.resize((256, 256), resample=Image.NEAREST))

def calculate_area_and_perimeter(mask):
    """计算二值化掩模的面积和周长"""
    # 寻找轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0
    
    # 合并所有轮廓
    combined_contour = np.vstack(contours)
    
    # 计算面积(像素单位)
    area = cv2.contourArea(combined_contour)
    
    # 计算周长(像素单位)
    perimeter = cv2.arcLength(combined_contour, closed=True)
    
    return area, perimeter

# 生成分割结果并计算肿瘤特征
def generate_segmentation(input_tensor, original_img):
    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # 为肝脏和肿瘤创建透明掩模(RGBA格式)
    liver_mask = np.zeros((*original_img.shape, 4), dtype=np.uint8)  # 4通道: RGBA
    tumor_mask = np.zeros((*original_img.shape, 4), dtype=np.uint8)  # 4通道: RGBA
    
    # 肝脏(绿色) - 包含类别1(肝脏)和3(重叠部分)
    liver_mask[pred_mask == 1] = [0, 255, 0, 255]
    liver_mask[pred_mask == 3] = [0, 255, 0, 255]    # 重叠部分视为肝脏
    
    # 肿瘤(红色) - 包含类别2(肿瘤)和3(重叠部分)
    tumor_mask[pred_mask == 2] = [255, 0, 0, 255]
    tumor_mask[pred_mask == 3] = [255, 0, 0, 255]    # 重叠部分视为肿瘤
    
    # 计算肿瘤面积和周长
    tumor_binary = (pred_mask == 2) | (pred_mask == 3)  # 肿瘤区域(包括重叠部分)
    tumor_area, tumor_perimeter = calculate_area_and_perimeter(tumor_binary.astype(np.uint8))
    
    # 将原始图像转换为RGB
    original_img_rgb = np.zeros((*original_img.shape, 3), dtype=np.uint8)
    original_img_rgb[..., 0] = original_img  # R
    original_img_rgb[..., 1] = original_img  # G
    original_img_rgb[..., 2] = original_img  # B
    
    # 创建叠加后的分割结果图像
    # 1. 将原始图像转换为float类型
    overlay_img = original_img_rgb.astype(np.float32) / 255.0
    
    # 2. 将肝脏遮罩叠加到原始图像上
    liver_alpha = liver_mask[..., 3:].astype(np.float32) / 255.0
    liver_rgb = liver_mask[..., :3].astype(np.float32) / 255.0
    overlay_img = overlay_img * (1 - liver_alpha) + liver_rgb * liver_alpha
    
    # 3. 将肿瘤遮罩叠加到已经包含肝脏的图像上
    tumor_alpha = tumor_mask[..., 3:].astype(np.float32) / 255.0
    tumor_rgb = tumor_mask[..., :3].astype(np.float32) / 255.0
    overlay_img = overlay_img * (1 - tumor_alpha) + tumor_rgb * tumor_alpha
    
    # 4. 转换回uint8
    overlay_img = (overlay_img * 255).astype(np.uint8)
    
    elapsed_time = time.time() - start_time
    
    # 准备结果字典
    result = {
        "original": original_img_rgb,
        "overlay": overlay_img,  # 新增叠加后的图像
        "liver_mask": liver_mask,
        "tumor_mask": tumor_mask,
        "tumor_area": tumor_area,
        "tumor_perimeter": tumor_perimeter,
        "message": "分割完成",
        "time_elapsed": f"{elapsed_time:.3f}s"
    }
    return result

def image_to_base64(img_array):
    """将numpy数组图像转换为base64字符串"""
    # 检查图像通道数
    if img_array.shape[-1] == 3:  # RGB图像
        img = Image.fromarray(img_array, 'RGB')
    elif img_array.shape[-1] == 4:  # RGBA图像
        img = Image.fromarray(img_array, 'RGBA')
    else:  # 灰度图像
        img = Image.fromarray(img_array, 'L')
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return """
    <h1>医学图像分割API</h1>
    <p>可用端点:</p>
    <ul>
        <li>POST /segment - 上传医学图像</li>
    </ul>
    """

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 保存临时文件
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    try:
        # 预处理图像
        input_tensor, original_img = preprocess_image(temp_path)
        
        # 生成分割结果
        result = generate_segmentation(input_tensor, original_img)
        
        # 将所有图像转换为base64
        response_data = {
            "status": "success",
            "images": {
                "original": image_to_base64(result["original"]),
                "overlay": image_to_base64(result["overlay"]),  # 确保包含叠加图像
                "liver_mask": image_to_base64(result["liver_mask"]),
                "tumor_mask": image_to_base64(result["tumor_mask"])
            },
            "tumor_metrics": {
                "area": result["tumor_area"],
                "perimeter": result["tumor_perimeter"]
            },
            "message": result["message"],
            "time_elapsed": result["time_elapsed"]
        }
        
        # 删除临时文件
        os.remove(temp_path)
        
        return jsonify(response_data)
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "分割失败"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)