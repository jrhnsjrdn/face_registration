import threading
import time

import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

def camera_loop():
    global frame, running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while running:
        ret, f = cap.read()
        if ret:
            frame = f
        time.sleep(0.001)  # biar CPU ga 100%

    cap.release()

def start_camera():
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

def get_frame():
    ret, frame = cap.read()
    if not ret: return None
    return frame

def stop_camera():
    global running
    running = False
