from flask import Flask, render_template, request, Response, send_file
from ultralytics import YOLO
import cv2
import threading
import time
import os
from datetime import datetime
import queue

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== YOLO Model (Optimized) =====
model = YOLO("yolov8n.pt").cuda()

# Optional warm-up
dummy_path = "static/dummy.jpg"
if os.path.exists(dummy_path):
    dummy = cv2.imread(dummy_path)
    if dummy is not None:
        model(dummy)

# ===== Global Variables for Threaded Frame Handling =====
frame_queue = queue.Queue(maxsize=1)
display_frame = None
camera_running = True

# ===== Webcam Capture Thread =====
def capture_thread():
    global camera_running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)

    cap.release()
    print("[INFO] Camera released")

# ===== YOLO Detection Thread =====
def detection_thread():
    global display_frame
    frame_count = 0

    while camera_running:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        if frame_count % 3 == 0:  # YOLO every 3rd frame
            results = model(frame, imgsz=96, conf=0.6, half=True, device=0)
            annotated = results[0].plot()
            display_frame = annotated
        else:
            display_frame = frame

        frame_count += 1

# ===== Flask Frame Generator =====
def generate_frames():
    global display_frame
    while True:
        if display_frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ===== Flask Routes =====
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)
    results[0].save(filename="static/result.jpg")
    return send_file("static/result.jpg", mimetype='image/jpeg')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== Main Entry Point =====
if __name__ == '__main__':
    # Start threads
    threading.Thread(target=capture_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()

    app.run(debug=True, threaded=True)
    camera_running = False
