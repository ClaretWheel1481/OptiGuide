import base64
import io
from flask import Flask
from flask_socketio import SocketIO, emit
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO('yolov12n.pt')

@socketio.on('connect')
def on_connect():
    print("客户端已连接")
    emit('message', {'data': '连接成功'})

@socketio.on('disconnect')
def on_disconnect():
    print("客户端已断开连接")


@socketio.on('image')
def handle_image(data):
    """
    前端需要发送 Base64 编码后的图像字符串。
    检测流程：
      1. 解码 Base64 数据，生成 PIL Image 对象
      2. 调用 YOLO 模型进行检测
      3. 将检测结果转换为 JSON 格式并返回前端
    """
    try:
        image_data = data.get("image", "")
        if image_data.startswith("data:image"):
            header, image_data = image_data.split(",", 1)
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        results = model(image)

        detections = results.pandas().xyxy[0].to_dict(orient="records")

        emit('detections', {'detections': detections})
    except Exception as e:
        print("检测过程中发生错误:", e)
        emit('error', {'error': str(e)})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
