import cv2
import threading
import time

frame = None
running = False
cap = None

def camera_loop():
    global frame, running, cap

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while running:
        ret, f = cap.read()
        if ret:
            frame = f

        time.sleep(0.001)  # biar CPU tidak penuh

    if cap is not None:
        cap.release()


def start_camera():
    global running
    if running:  # kalau sudah jalan, jangan start lagi
        return
    running = True
    threading.Thread(target=camera_loop, daemon=True).start()


def get_frame():
    global frame
    return frame


def stop_camera():
    global running, cap
    running = False
    time.sleep(0.1)
    if cap is not None:
        cap.release()
