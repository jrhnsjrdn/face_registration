import cv2
import face_recognition
import numpy as np

def encode_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")

    if len(locs) != 1:
        return None, None

    enc = face_recognition.face_encodings(rgb, locs)

    if len(enc) == 0:
        return None, None

    return enc[0], locs[0]


def recognize_faces(rgb_frame, known_encodings, known_names, known_guest_counts):
    locs = face_recognition.face_locations(rgb_frame, model="hog")
    encs = face_recognition.face_encodings(rgb_frame, locs)

    names, guests = [], []
    for enc in encs:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        dist = face_recognition.face_distance(known_encodings, enc)

        idx = np.argmin(dist) if len(dist) else None

        if idx is not None and matches[idx]:
            names.append(known_names[idx])
            guests.append(known_guest_counts[idx])
        else:
            names.append("Unknown")
            guests.append(0)

    return locs, names, guests
