import cv2
import os
import time
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# 确保图片存储目录存在
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 加载 YOLOv8 目标检测模型（使用 nano 版本以适应树莓派算力）
model = YOLO('yolov8n.pt')

# 初始化摄像头
camera = cv2.VideoCapture(0)
# 降低分辨率以提升树莓派上的视频流畅度
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    """读取摄像头帧并编码为 JPEG 视频流"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由，供前端 <img> 标签调用"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """拍照接口：读取当前帧并保存"""
    success, frame = camera.read()
    if success:
        timestamp = int(time.time())
        filename = f"{timestamp}.jpg"
        filepath = os.path.join('static/images', filename)
        cv2.imwrite(filepath, frame)
        return jsonify({"status": "success", "image_url": f"/static/images/{filename}", "filename": filename})
    return jsonify({"status": "error", "message": "无法读取摄像头画面"})

@app.route('/recognize', methods=['POST'])
def recognize():
    """识别接口：对刚才拍摄的图片进行 YOLO 识别"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
         return jsonify({"status": "error", "message": "未提供文件名"})

    source_path = os.path.join('static/images', filename)
    result_path = os.path.join('static/results', f"res_{filename}")

    # 调用 YOLO 进行推理
    results = model(source_path)
    
    # 绘制标注框并保存结果
    annotated_frame = results[0].plot()
    cv2.imwrite(result_path, annotated_frame)

    return jsonify({"status": "success", "result_url": f"/static/results/res_{filename}"})

if __name__ == '__main__':
    # 在局域网内公开访问，开启多线程模式
    app.run(host='0.0.0.0', port=5000, threaded=True)