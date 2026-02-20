import cv2
import os
import time
import numpy as np
import onnxruntime as ort
import psutil
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

os.makedirs('static/images', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 初始化摄像头
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------- ONNX 推理初始化 -----------------
model_path = 'yolov8n.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog']

# 实时识别开关标记
realtime_enabled = False

def process_frame_with_yolo(img):
    """封装核心视觉识别与绘框逻辑"""
    original_h, original_w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    
    outputs = session.run(None, {input_name: blob})[0]
    outputs = np.squeeze(outputs).T
    
    boxes, scores, class_ids = [], [], []
    x_factor = original_w / 640.0
    y_factor = original_h / 640.0

    for row in outputs:
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        if max_score > 0.5:
            class_id = np.argmax(classes_scores)
            x, y, w, h = row[0], row[1], row[2], row[3]
            left = int((x - w/2) * x_factor)
            top = int((y - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            label = f"{CLASSES[class_ids[i]] if class_ids[i] < len(CLASSES) else class_ids[i]}: {scores[i]:.2f}"
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return img

def generate_frames():
    """视频流生成器，包含性能数据 OSD (On-Screen Display)"""
    global realtime_enabled
    prev_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        # 1. 如果开启了实时识别，执行 YOLO 推理
        if realtime_enabled:
            frame = process_frame_with_yolo(frame)
            
        # 2. 计算当前视频帧率 (FPS)
        current_time = time.time()
        # 加上 1e-5 防止极快速度下除以零报错
        fps = 1.0 / (current_time - prev_time + 1e-5) 
        prev_time = current_time
        
        # 3. 获取 CPU 和 内存使用率 (调用 psutil)
        # interval=None 表示非阻塞获取即时 CPU 使用率
        cpu_usage = psutil.cpu_percent(interval=None) 
        ram_usage = psutil.virtual_memory().percent
        
        # 4. 在画面左上角绘制性能信息 (OpenCV putText)
        # 格式：cv2.putText(图像, 文字, 坐标, 字体, 缩放, 颜色(BGR), 线宽)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"CPU: {cpu_usage}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"RAM: {ram_usage}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_realtime', methods=['POST'])
def toggle_realtime():
    """控制实时识别开关的接口"""
    global realtime_enabled
    data = request.get_json()
    realtime_enabled = data.get('enabled', False)
    return jsonify({"status": "success", "realtime_enabled": realtime_enabled})

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        filename = f"{int(time.time())}.jpg"
        filepath = os.path.join('static/images', filename)
        cv2.imwrite(filepath, frame)
        return jsonify({"status": "success", "image_url": f"/static/images/{filename}", "filename": filename})
    return jsonify({"status": "error", "message": "无法读取摄像头画面"})

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    filename = data.get('filename')
    if not filename: return jsonify({"status": "error"})

    source_path = os.path.join('static/images', filename)
    result_path = os.path.join('static/results', f"res_{filename}")

    img = cv2.imread(source_path)
    # 复用之前封装好的函数
    result_img = process_frame_with_yolo(img)
    cv2.imwrite(result_path, result_img)

    return jsonify({"status": "success", "result_url": f"/static/results/res_{filename}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)