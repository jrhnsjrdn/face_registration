import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

def get_frame():
    ret, frame = cap.read()
    if not ret: return None
    return frame

def release_cam():
    cap.release()
