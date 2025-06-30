from flask import Flask, render_template, request, Response
import cv2
from ultralytics import YOLO
import os
import uuid
import numpy as np

app = Flask(__name__)
model = YOLO("best.pt")

labels = ['bread_coconut', 'bread_roll', 'chiffon', 'cookie', 'croissant', 'croissant_chocolate',
          'donut', 'mochi', 'mooncake', 'muffin', 'sandwich', 'tart_egg']

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Kiểm tra có đang chạy trên Render không
IS_RENDER = os.environ.get("RENDER") == "true"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return "✅ Flask app is running properly."

# === CAMERA STREAM (CHỈ DÙNG KHI LOCAL) ===
if not IS_RENDER:
    def generate_camera_stream():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            frame = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(frame, "⚠️ Cannot access camera.", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                results = model.predict(frame, conf=0.5)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = box
                    label = labels[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            cap.release()

    @app.route('/camera')
    def camera():
        return render_template('camera_result.html')

    @app.route('/camera-feed')
    def camera_feed():
        return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === IMAGE DETECTION ===
@app.route('/detect-image', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            result_path = os.path.join(RESULT_FOLDER, filename)
            file.save(input_path)

            results = model.predict(input_path, conf=0.5)[0]
            result_img = results.plot()

            label_counts = {label: 0 for label in labels}
            for cls_id in results.boxes.cls.cpu().numpy().astype(int):
                label = labels[cls_id]
                label_counts[label] += 1

            y_offset = 25
            for label, count in label_counts.items():
                if count > 0:
                    text = f"{label}: {count}"
                    cv2.putText(result_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    y_offset += 25

            cv2.imwrite(result_path, result_img)

            filtered_counts = {k: v for k, v in label_counts.items() if v > 0}

            return render_template('image_result.html',
                                   result_image=result_path,
                                   label_counts=filtered_counts,
                                   all_labels=labels)
    return render_template('detect_image.html')

# === VIDEO DETECTION ===
@app.route('/detect-video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            input_filename = f"{uuid.uuid4().hex}.mp4"
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            file.save(input_path)

            output_filename = f"processed_{input_filename}"
            output_path = os.path.join(RESULT_FOLDER, output_filename)

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            last_frame_counts = {label: 0 for label in labels}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=0.5, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()

                frame_counts = {label: 0 for label in labels}
                for cls_id in class_ids:
                    label = labels[cls_id]
                    frame_counts[label] += 1

                last_frame_counts = frame_counts

                for box, cls_id, conf in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = box
                    label = labels[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                y_offset = 30
                for label, count in frame_counts.items():
                    if count > 0:
                        cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        y_offset += 30

                out.write(frame)

            cap.release()
            out.release()

            filtered_counts = {k: v for k, v in last_frame_counts.items() if v > 0}

            return render_template('video_result.html',
                                   video_file=f'results/{output_filename}',
                                   label_counts=filtered_counts,
                                   all_labels=labels)
    return render_template('detect_video.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render cung cấp PORT động
    app.run(host="0.0.0.0", port=port)        # Flask phải lắng nghe 0.0.0.0

