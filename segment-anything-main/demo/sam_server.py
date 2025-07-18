from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import matplotlib.pyplot as plt
import base64
import io
import time
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

app = Flask(__name__)
CORS(app)

# 初始化SAM模型
print("正在初始化SAM模型...")
sam_checkpoint = r"D:\VSpython\segment-anything-main\model\sam_vit_h_4b8939.pth"  # 修改为你的模型路径
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def show_mask(mask, ax=None, random_color=False):
    """将掩码转为 RGBA 格式，掩码区域为绿色，其他区域透明"""
    h, w = mask.shape[-2:]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)  # 全透明背景

    # 掩码区域：绿色 (0,255,0)，透明度 0.6 (约 153)
    rgba[mask > 0] = np.array([0, 255, 0, 153], dtype=np.uint8)

    return rgba

def image_to_base64(img_array):
    """将 numpy 图像（支持 RGBA）转为 base64"""
    # 如果是浮点型，转为 uint8
    if img_array.dtype in [np.float32, np.float64]:
        img_array = (img_array * 255).astype(np.uint8)

    # 单通道变 3 通道
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # 读取图像
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        start_time = time.time()

        # 预测
        predictor.set_image(image_np)

        # 示例：默认使用固定点
        input_point = np.array([[150, 150]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # 取得分最低的掩码
        worst_idx = np.argmin(scores)
        best_mask = masks[worst_idx]

        # 生成 RGBA 格式的透明遮罩
        mask_image = show_mask(best_mask)

        # 生成叠加图像
        overlay = image_np.copy()
        overlay[best_mask] = (overlay[best_mask] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)

        elapsed_time = time.time() - start_time

        # 返回结果
        response = {
            "status": "success",
            "images": {
                "original": image_to_base64(image_np),  # 原图
                "mask": image_to_base64(mask_image),    # 透明背景遮罩
                "overlay": image_to_base64(overlay)     # 直接叠加的版本（可选）
            },
            "metrics": {
                "score": float(scores[worst_idx]),
                "time_elapsed": f"{elapsed_time:.3f}s"
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Segmentation failed"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
