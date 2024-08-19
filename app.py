import cv2
import numpy as np
from flask import Flask, render_template, Response, request
import torch
import threading

app = Flask(__name__)

# Load YOLOv5 model
modelo = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Global variable to control object detection
run_detection = False

def detect_objects():
    global run_detection
    # Change the index to 1 to use the USB camera
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and run_detection:
        _, frame = cap.read()
        salida = modelo(frame)
        annotated_frame = np.squeeze(salida.render())
        _, jpeg = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if run_detection:
        return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Detection is not running."

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global run_detection
    run_detection = True
    # Start detection in a new thread to avoid blocking the Flask server
    threading.Thread(target=detect_objects).start()
    return "Detection started."

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global run_detection
    run_detection = False
    return "Detection stopped."

if __name__ == '__main__':
    app.run(debug=True)
