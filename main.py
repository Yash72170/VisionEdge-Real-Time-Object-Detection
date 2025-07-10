from flask import Flask, render_template, request, Response, send_file
from ultralytics import YOLO
import cv2
import os
from datetime import datetime

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Make sure this file is in your root folder

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Homepage
@app.route('/')
def index():
    return render_template("index.html")

# Handle image upload + detection
@app.route('/upload', methods=["POST"])
def upload():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    # Save file with timestamp
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run YOLOv8 detection
    results = model(filepath)
    results[0].save(filename="static/result.jpg")

    return send_file("static/result.jpg", mimetype='image/jpeg')

# Live webcam frame generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open default webcam
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # YOLO detection on frame
            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield as multipart stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        print("Webcam released")

# Webcam route
@app.route('/video')
def video():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)