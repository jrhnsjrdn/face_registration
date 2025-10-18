import cv2
import face_recognition
import numpy as np
import sqlite3
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

# --- Database setup ---
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS registered_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding TEXT NOT NULL
)
''')
conn.commit()

# --- Load registered faces ---
def load_registered_faces():
    cursor.execute("SELECT name, encoding FROM registered_faces")
    rows = cursor.fetchall()
    known_encodings = []
    known_names = []

    for name, encoding_json in rows:
        encoding = np.array(json.loads(encoding_json))
        known_encodings.append(encoding)
        known_names.append(name)

    print(f"[INFO] Loaded {len(known_names)} registered faces.")
    return known_encodings, known_names

known_encodings, known_names = load_registered_faces()

# --- Save new face ---
def save_face_to_db(name, encoding):
    encoding_json = json.dumps(encoding.tolist())
    cursor.execute("INSERT INTO registered_faces (name, encoding) VALUES (?, ?)", (name, encoding_json))
    conn.commit()
    print(f"[INFO] Face '{name}' saved to database.")
    reload_registered_faces()

def reload_registered_faces():
    global known_encodings, known_names
    known_encodings, known_names = load_registered_faces()

# --- GUI setup ---
root = tk.Tk()
root.title("Face Registration & Recognition System (Smooth)")
root.geometry("800x600")

video_label = tk.Label(root)
video_label.pack()

name_label = tk.Label(root, text="Nama:")
name_label.pack(pady=5)
name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=5)

capture_btn = tk.Button(root, text="📸 Capture & Simpan", font=("Arial", 12), bg="#4CAF50", fg="white")
capture_btn.pack(pady=10)

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
frame_count = 0
process_every_n_frames = 10  # only recognize every N frames

# --- Global variables ---
face_locations = []
face_names = []
processing_thread = None
lock = threading.Lock()

# --- Background face recognition function ---
def process_face_recognition(rgb_small_frame):
    global face_locations, face_names

    small_face_locations = face_recognition.face_locations(rgb_small_frame)
    small_face_encodings = face_recognition.face_encodings(rgb_small_frame, small_face_locations)

    names = []
    for face_encoding in small_face_encodings:
        if len(known_encodings) == 0:
            names.append("Unknown")
            continue

        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            names.append(known_names[best_match_index])
        else:
            names.append("Unknown")

    with lock:
        face_locations = small_face_locations
        face_names[:] = names  # replace content

# --- Show camera frame ---
def show_frame():
    global frame_count, processing_thread

    ret, frame = cap.read()
    if not ret:
        return

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    frame_count += 1

    # Launch recognition in background thread
    if frame_count % process_every_n_frames == 0:
        if processing_thread is None or not processing_thread.is_alive():
            processing_thread = threading.Thread(target=process_face_recognition, args=(rgb_small_frame,))
            processing_thread.daemon = True
            processing_thread.start()

    # Draw boxes and names from last processed frame
    with lock:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up to full frame size
            top = int(top / 0.4)
            right = int(right / 0.4)
            bottom = int(bottom / 0.4)
            left = int(left / 0.4)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert to Tkinter Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

show_frame()

# --- Capture and save face ---
def capture_and_save():
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("Input Kosong", "Masukkan nama terlebih dahulu!")
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Gagal menangkap frame dari kamera.")
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations_now = face_recognition.face_locations(rgb_frame)
    if not face_locations_now:
        messagebox.showwarning("Tidak Ada Wajah", "Tidak ada wajah terdeteksi!")
        return

    encodings = face_recognition.face_encodings(rgb_frame, face_locations_now)
    if not encodings:
        messagebox.showerror("Error", "Gagal membuat encoding wajah.")
        return

    save_face_to_db(name, encodings[0])
    messagebox.showinfo("Sukses", f"Wajah '{name}' berhasil disimpan ke database!")
    name_entry.delete(0, tk.END)

capture_btn.config(command=capture_and_save)

# --- Safely close window ---
def on_closing():
    cap.release()
    conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
