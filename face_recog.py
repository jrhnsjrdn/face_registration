import cv2
import face_recognition
import numpy as np


def encode_face(frame):
    # Convert ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) != 1:
        return None, None  # Harus 1 wajah saja

    # Ambil encoding
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(encodings) == 0:
        return None, None

    return encodings[0], face_locations[0]


def recognize_faces(rgb_small_frame, known_encodings, known_names, known_guest_counts):
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names, guests = [], []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        best_match_index = np.argmin(face_distances) if len(face_distances) else None

        if best_match_index is not None and matches[best_match_index]:
            names.append(known_names[best_match_index])
            guests.append(known_guest_counts[best_match_index])
        else:
            names.append("Unknown")
            guests.append(0)

    return face_locations, names, guests
