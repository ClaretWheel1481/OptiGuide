import cv2
import subprocess
import numpy as np


def main():
    # RTMP流地址（需与Flutter前端一致）
    rtmp_url = "rtmp://192.168.6.120:1935/live/stream"

    # 方法1：直接使用OpenCV（需OpenCV支持FFmpeg）
    try:
        cap = cv2.VideoCapture(rtmp_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧，检查流地址是否正确！")
                break
            cv2.imshow("Real-time Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    except Exception as e:
        print(f"OpenCV直接读取失败：{e}\n尝试使用FFmpeg管道方法...")

        # 方法2：通过FFmpeg管道读取（通用方法）
        width, height = 640, 480  # 假设前端推流分辨率为640x480
        command = [
            'ffmpeg',
            '-i', rtmp_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)

        while True:
            raw_frame = pipe.stdout.read(width * height * 3)
            if len(raw_frame) == 0:
                break
            frame = np.frombuffer(raw_frame, dtype='uint8').reshape((height, width, 3))
            cv2.imshow("Real-time Stream (FFmpeg)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pipe.terminate()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()