# face_worker.py
from face_recog import recognize_faces
import cv2
import numpy as np
import time

def worker(frame_queue, result_queue, known_encodings_serial, known_names, known_guests):
    """
    frame_queue/result_queue are multiprocessing.Queue instances.
    known_encodings_serial: list of lists (serializable) - each encoding is list of floats
    known_names, known_guests: lists
    This worker blocks on frame_queue.get() and pushes results to result_queue.
    """
    # convert known_encodings_serial to numpy arrays for faster ops
    known_encodings = [np.array(e) for e in known_encodings_serial]

    while True:
        try:
            frame = frame_queue.get()  # blocking
            if frame is None:
                time.sleep(0.01)
                continue

            # resize and convert to RGB for faster processing
            small = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs, names, guests = recognize_faces(rgb, known_encodings, known_names, known_guests)
            # push result (locs are in small-frame coords)
            try:
                if not result_queue.full():
                    result_queue.put((locs, names, guests))
            except Exception:
                pass

        except Exception as e:
            # keep worker alive on error
            print("[WORKER ERROR]", e)
            time.sleep(0.05)
