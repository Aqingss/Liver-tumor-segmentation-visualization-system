import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from model import UNet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载模型（灰度输入）
model = UNet(in_channels=1, num_classes=4)
model.load_state_dict(torch.load('models_weight/best.pth', map_location=device))
model.to(device)
model.eval()

# 3. 加载灰度图像
img_path = 'volume-125_219.png'
img = Image.open(img_path).convert('L')
original_img = np.array(img.resize((256, 256), resample=Image.NEAREST))  # 保存原图像做可视化
original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)  # 转换成 3 通道

# 4. 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0).to(device)

# 5. 推理
with torch.no_grad():
    output = model(input_tensor)  # [1, 4, H, W]
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # [H, W]
    probs = F.softmax(output, dim=1)
    center_probs = probs[0, :, 128, 128].cpu().numpy()
    print("中心像素预测类别概率分布：", center_probs)
    print("预测结果中出现的类别：", np.unique(pred_mask))

# 6. 生成彩色掩码（只突出非背景区域）
# 类别颜色映射：背景(0)=黑，肝脏(1)=绿色，肿瘤(2)=红色，重叠(3)=蓝色
colormap = {
    1: (0, 255, 0),   # Green for liver
    2: (255, 0, 0),   # Red for tumor
    3: (0, 0, 255),   # Blue for overlap
}

mask_color = np.zeros_like(original_img_color)

for cls_id, color in colormap.items():
    mask_color[pred_mask == cls_id] = color

# 7. 叠加到原图上（透明融合）
alpha = 0.5
overlay = cv2.addWeighted(original_img_color, 1 - alpha, mask_color, alpha, 0)

# 8. 显示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Overlay Segmentation")
plt.imshow(overlay[..., ::-1])  # OpenCV 是 BGR，要转为 RGB
plt.axis('off')

plt.tight_layout()
plt.show()
