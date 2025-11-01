import face_recognition
import numpy as np


def encode_face(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return None, None

    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return encodings[0], face_locations


def recognize_faces(rgb_small_frame, known_encodings, known_names, known_guest_counts):
    face_locations = face_recognition.face_locations(rgb_small_frame)
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
