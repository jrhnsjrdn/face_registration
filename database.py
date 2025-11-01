import sqlite3
import json
import numpy as np

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS registered_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding TEXT NOT NULL,
        guest_count INTEGER DEFAULT 1
    )
    """)
    conn.commit()
    conn.close()

def load_registered_faces():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding, guest_count FROM registered_faces")
    rows = cursor.fetchall()
    conn.close()

    known_encodings, known_names, known_guest_counts = [], [], []

    for name, encoding_json, guest_count in rows:
        encoding = np.array(json.loads(encoding_json))
        known_encodings.append(encoding)
        known_names.append(name)
        known_guest_counts.append(guest_count)

    print(f"[DB] Loaded {len(known_names)} faces")
    return known_encodings, known_names, known_guest_counts

def save_face_to_db(name, guest_count, encoding):
    encoding_json = json.dumps(encoding.tolist())
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO registered_faces (name, encoding, guest_count) VALUES (?, ?, ?)",
        (name, encoding_json, guest_count)
    )
    conn.commit()
    conn.close()
    print(f"[DB] Saved face {name}, guest {guest_count}")
