# 医学图像分割可视化平台

## 📌 项目简介

本项目基于深度学习与医学图像处理技术，实现了 **肝脏及肿瘤自动分割** 的可视化交互平台。  
系统支持 **多模型对比**、**性能指标可视化** 与 **交互式结果浏览**，为医学诊断提供辅助支持。

包含三种后端模型服务：

- **Attention U-Net**（更精准的小肿瘤分割）
- **U-Net**（经典医学图像分割）
- **SAM**（快速肝脏区域定位）

以及一个基于 **Web 前端** 的交互平台。

---

## 📂 目录结构

├── Attention-UNet/ # Attention U-Net 模型服务

│ └── Attention_UNet.py

├── Liver_UNet/ # U-Net 模型服务

│ └── unet_server.py

├── segment-anything-main/ # SAM 模型服务

│ └── sam_server.py

└── liver/ # Web 可视化前端

├── index.html

└── （其余前端源码文件）



---

## 🚀 启动步骤

### 1. 启动后端服务

分别进入对应目录并启动服务（建议使用不同终端窗口）：

```bash
# 1）Attention U-Net 服务（5002端口）
cd Attention-UNet
python Attention_UNet.py

# 2）U-Net 服务（5000端口）
cd ../Liver_UNet
python unet_server.py

# 3）SAM 服务（5001端口）
cd ../segment-anything-main
python sam_server.py
```
确保三个服务全部正常启动后，再执行前端部分。

### 2. 启动前端可视化平台
```bash
cd ../liver
npm install   # 首次运行需安装依赖
npm run dev   # 启动本地开发服务器
```
在浏览器中打开提示网址（如：http://localhost:5173 或 http://127.0.0.1:5173）即可访问系统。

## 💡 功能概述
-数据输入：支持上传单张 2D 医学影像或 2.5D 模式的整组切片

-原始图像展示：支持分页浏览影像序列

-模型设置：自由选择三种模型并可调整参数

-分割结果展示：肝脏（绿色）/肿瘤（红色）透明遮罩叠加，支持透明度调节

-性能指标展示：自动生成 Dice、IoU 等指标数值卡片及曲线图

-方法对比：支持同模型参数对比及跨模型结果对比

## 🔧 环境依赖
Python 3.8+

PyTorch >= 1.10

Flask

Node.js & npm

## 📝 备注
后端服务需保持运行状态，否则前端无法获取分割结果

如需修改端口，请同步修改前端 fetch 请求中的 URL
