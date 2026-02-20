# 🍓 树莓派轻量级 Web 目标检测平台 (Raspberry Pi YOLO Vision)

本项目是一个基于 **树莓派 4B** 的机器视觉边缘计算工程实践。通过 Flask 框架搭建轻量级 Web 服务，结合罗技 C670i 摄像头实现局域网内的实时视频流监控、抓拍及基于 YOLOv8 的目标检测。

为解决 ARM 架构下的 `Illegal instruction` 报错及算力瓶颈，本项目摒弃了臃肿的原生 PyTorch 框架，采用 **ONNX Runtime** 进行模型推理，大幅提升了边缘设备上的运行帧率。

## ✨ 功能特性

1. **实时图传**：局域网内可通过浏览器访问前端网页，实时查看摄像头视频流。
2. **异步抓拍**：点击按钮即可调用前端 API 保存当前视频帧。
3. **目标检测**：接入 YOLOv8n 模型，对抓拍图像进行前向推理，绘制 Bounding Box 并返回预览。
4. **历史记录**：自动按时间戳保存所有原始抓拍图像与带有检测标注的识别结果。

## 🛠️ 硬件与开发环境

* **边缘计算设备**：Raspberry Pi 4B (64-bit OS, aarch64)
* **视觉采集设备**：Logitech C670i USB 摄像头 (`/dev/video0`)
* **本地开发机**：Mac (Intel 芯片)
* **核心技术栈**：Python, Flask, OpenCV, ONNX Runtime, HTML/JS

---

## 🚀 部署全指南

### 阶段一：本地模型准备 (在 Mac 侧执行)

由于直接在树莓派上运行原生 YOLO 模型不仅极度消耗资源，且容易引发指令集兼容问题，我们需要在本地 Mac 上先将模型转换为轻量级的 ONNX 格式。

1. **配置隔离的 Conda 环境**（避免 Python 3.13 依赖地狱）：
```bash
conda init zsh
source ~/.zshrc
conda create -n cv_env python=3.10 -y
conda activate cv_env

```


2. **安装依赖并导出 ONNX 模型**：
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx

```


*执行完毕后，当前目录下会生成 `yolov8n.onnx` 文件。*

### 阶段二：代码推送到树莓派

推荐使用 **VS Code Remote-SSH** 插件进行部署：

1. 通过 SSH 连接至树莓派：`ssh pi@<树莓派IP>`
2. 在树莓派创建项目目录：`mkdir ~/pi_vision`
3. 使用 VS Code 直接打开该远端目录。
4. 将本地编写好的 `app.py`、`templates/index.html` 以及刚才导出的 `yolov8n.onnx` 直接拖拽到 VS Code 目录中。

### 阶段三：树莓派环境配置 (在树莓派 SSH 终端执行)

1. **创建并激活 Python 虚拟环境**：
```bash
cd ~/pi_vision
python3 -m venv env
source env/bin/activate

```


2. **安装轻量级推理依赖**（完全卸载/跳过 PyTorch）：
```bash
pip install --upgrade pip
pip install flask onnxruntime opencv-python-headless numpy

```



### 阶段四：启动服务

1. 确保摄像头已正确插入树莓派 USB 接口（可通过 `ls /dev/video*` 确认）。
2. 在虚拟环境中运行后端服务：
```bash
python app.py

```


3. **访问平台**：在同一局域网下的任意电脑或手机浏览器中，访问 `http://<树莓派的局域网IP>:5000` 即可开始使用。

---

## 📂 项目文件结构

```text
pi_vision/
├── app.py                 # Flask 后端核心逻辑与 ONNX 推理代码
├── yolov8n.onnx           # 转换后的轻量级 YOLO 权重文件
├── env/                   # 树莓派 Python 虚拟环境（不入库）
├── templates/
│   └── index.html         # 前端交互页面
└── static/
    ├── images/            # 抓拍原始图像存储路径
    └── results/           # 识别后图像存储路径

```

## 📝 踩坑记录与解决方案

* **问题**：在树莓派安装 `ultralytics` 运行测试时抛出 `Illegal instruction`。
* **原因**：预编译的 PyTorch wheel 包包含了 Cortex-A72 (ARMv8.0-A) 不支持的 ARMv8.2-A 高级指令集。
* **解决**：彻底卸载 PyTorch 环境，在开发机上完成模型导出，树莓派端仅保留 `onnxruntime` 及 OpenCV 的 DNN 模块进行纯粹的前向推理和后处理（NMS），不仅解决了崩溃问题，还显著降低了内存占用。

