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
        name TEXT NOT NULL UNIQUE,
        encoding TEXT NOT NULL,
        guest_count INTEGER DEFAULT 1,
        checkin_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def get_all_faces():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding, guest_count FROM registered_faces")
    rows = cursor.fetchall()
    conn.close()

    faces = []
    for name, encoding_json, guest_count in rows:
        encoding = np.array(json.loads(encoding_json))
        faces.append({
            "name": name,
            "embedding": encoding,
            "guest_count": guest_count
        })
    return faces


def load_registered_faces():
    # Legacy usage (opsional kalau dipake di worker)
    faces = get_all_faces()
    known_encodings = [x["embedding"] for x in faces]
    known_names = [x["name"] for x in faces]
    known_guest_counts = [x["guest_count"] for x in faces]

    print(f"[DB] Loaded {len(known_names)} faces")
    return known_encodings, known_names, known_guest_counts


def save_face_to_db(name, guest_count, encoding):
    encoding_json = json.dumps(encoding.tolist())
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR REPLACE INTO registered_faces (name, encoding, guest_count) VALUES (?, ?, ?)",
        (name, encoding_json, guest_count)
    )

    conn.commit()
    conn.close()
    print(f"[DB] Saved face {name}, guest {guest_count}")


def get_dashboard_stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), SUM(guest_count) FROM registered_faces")
    row = cursor.fetchone()
    conn.close()

    total_registered = row[0] or 0
    total_guests = row[1] or 0

    return total_registered, total_guests
