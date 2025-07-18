from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. 加载掩码图像
mask = Image.open("prepared_dataset/test/masks/volume-62_111.png")
mask_np = np.array(mask)

# 2. 输出信息
print("非背景像素个数:", np.count_nonzero(mask_np))
print("唯一像素值:", np.unique(mask_np))  # 例如 [0 1 3] 表示背景、肝脏、重叠

# 3. 自定义颜色映射（0=背景，1=肝脏，2=肿瘤，3=重叠）
# 用 RGB 值表示每个类
cmap = mcolors.ListedColormap([
    (0.0, 0.0, 0.0),      # 背景: 黑色
    (0.0, 1.0, 0.0),      # 肝脏: 绿色
    (1.0, 0.0, 0.0),      # 肿瘤: 红色
    (0.0, 0.0, 1.0),      # 重叠: 蓝色
])

# 设置颜色边界
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# 4. 可视化
plt.figure(figsize=(6, 6))
plt.title("Ground Truth Mask")
plt.imshow(mask_np, cmap=cmap, norm=norm)
plt.colorbar(ticks=[0, 1, 2, 3], label="Class")
plt.axis('off')
plt.show()
