# camera.py
import cv2
import threading
import time

# camera thread stores last_frame in module-global var
_last_frame = None
_running = False

def _camera_loop(device=0, fps_target=30, queue_putter=None):
    global _last_frame, _running
    cap = cv2.VideoCapture(device)
    # optional tuning:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FPS, fps_target)
    except Exception:
        pass

    _running = True
    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        _last_frame = frame
        # if a frame queue provided, try put non-blocking
        if queue_putter is not None:
            try:
                if not queue_putter.full():
                    queue_putter.put(frame)
            except Exception:
                pass

        # slight sleep to avoid busy-loop
        time.sleep(0.001)

    cap.release()

def start_camera(frame_queue=None, device=0, fps_target=30):
    t = threading.Thread(target=_camera_loop, args=(device, fps_target, frame_queue), daemon=True)
    t.start()
    return t

def get_frame():
    return _last_frame

def stop_camera():
    global _running
    _running = False
