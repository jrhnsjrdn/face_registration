import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
from database import save_face_to_db, load_registered_faces, get_dashboard_stats, init_db
from face_recog import encode_face, recognize_faces
from camera import get_frame, start_camera


class FaceApp:
    def __init__(self, root):
        self.on_close = None
        self.root = root
        self.root.title("Face Registration & Recognition")
        self.root.geometry("800x650")
        start_camera()

        # Init DB (IMPORTANT)
        init_db()

        # Dashboard Frame
        self.dashboard_frame = tk.Frame(root, pady=10)
        self.dashboard_frame.pack()

        self.total_people_label = tk.Label(self.dashboard_frame, text="Total Tamu Terdaftar: 0", font=("Arial", 12),
                                           fg="blue")
        self.total_people_label.grid(row=0, column=0, padx=10)

        self.total_guest_label = tk.Label(self.dashboard_frame, text="Total Undangan Keseluruhan: 0",
                                          font=("Arial", 12), fg="green")
        self.total_guest_label.grid(row=0, column=1, padx=10)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Input Form
        tk.Label(root, text="Nama Tamu:").pack()
        self.name_entry = tk.Entry(root, width=30)
        self.name_entry.pack()

        tk.Label(root, text="Jumlah Tamu:").pack()
        self.guest_entry = tk.Entry(root, width=10)
        self.guest_entry.insert(0, "1")
        self.guest_entry.pack()

        tk.Button(root, text="📸 Capture & Save", bg="green", fg="white",
                  command=self.capture_face).pack(pady=10)

        # Load face data
        self.known_encodings, self.known_names, self.known_guests = load_registered_faces()
        self.update_dashboard()

        # Start camera thread
        self.thread = threading.Thread(target=self.stream_camera, daemon=True)
        self.thread.start()

        # Proper exit handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_dashboard(self):
        total_people, total_guests = get_dashboard_stats()
        self.total_people_label.config(text=f"Total Tamu Terdaftar: {total_people}")
        self.total_guest_label.config(text=f"Total Tamu Undangan Keseluruhan: {total_guests}")

    def reload_faces(self):
        self.known_encodings, self.known_names, self.known_guests = load_registered_faces()

    def capture_face(self):
        frame = get_frame()
        if frame is None: return

        name = self.name_entry.get().strip()
        count = self.guest_entry.get().strip()

        if not name or not count.isdigit():
            messagebox.showwarning("Invalid", "Isi nama & jumlah tamu valid!")
            return

        encoding, face = encode_face(frame)
        if encoding is None:
            messagebox.showwarning("No Face", "Tidak ada wajah terdeteksi!")
            return

        save_face_to_db(name, int(count), encoding)
        self.root.after(500, self.reload_faces)
        self.root.after(500, self.update_dashboard)
        messagebox.showinfo("Success", "Wajah berhasil disimpan!")
        self.name_entry.delete(0, tk.END)
        self.guest_entry.delete(0, tk.END)
        self.guest_entry.insert(0, "1")

    def stream_camera(self):
        frame_skip = 0

        while True:
            try:
                frame = get_frame()
                if frame is None:
                    continue

                # scale kecil untuk speed recognition
                small = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                # recognition tiap 3 frame
                frame_skip += 1
                if frame_skip % 3 == 0:
                    locs, names, guests = recognize_faces(
                        rgb, self.known_encodings, self.known_names, self.known_guests
                    )
                    self.last_detect = (locs, names, guests)

                # kalau ada last result, gambar bounding box
                if hasattr(self, "last_detect"):
                    locs, names, guests = self.last_detect
                    for (top, right, bottom, left), name, g in zip(locs, names, guests):
                        top, right, bottom, left = int(top / 0.4), int(right / 0.4), int(bottom / 0.4), int(left / 0.4)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        label = f"{name} ({g})" if name != "Unknown" else "Unknown"
                        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.video_label.config(image=img)
                self.video_label.image = img

            except Exception as e:
                print("[ERROR STREAM]", e)
                continue
