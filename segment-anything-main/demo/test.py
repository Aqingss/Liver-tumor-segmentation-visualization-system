import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# 显示分割掩码
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# 显示标记点
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

# 显示边界框
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

print("正在加载图像...")
image = cv2.imread(r"D:\VSpython\Liver_UNet\volume-100_581.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("显示原始图像...")
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

print("正在初始化SAM模型...")
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = r"D:\VSpython\segment-anything-main\model\sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"  # or  "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

print("正在准备图像进行分割...")
predictor = SamPredictor(sam)
predictor.set_image(image)

print("定义分割提示点...")
# 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
input_point = np.array([[150, 150]])  # 标记点
input_label = np.array([1])  # 点所对应的标签

print("显示标记点位置...")
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

print("正在进行分割预测...")
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(f"预测完成，生成了{masks.shape[0]}个可能的分割掩码")  # (number_of_masks) x H x W

# 三个置信度不同的图
print("正在显示分割结果...")
for i, (mask, score) in enumerate(zip(masks, scores)):
    print(f"显示掩码 {i+1}/{len(masks)} (置信度: {score:.3f})...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

print("程序执行完毕!")