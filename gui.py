# gui.py
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import time
from multiprocessing import Process, Queue

from database import init_db, save_face_to_db, load_registered_faces, get_dashboard_stats
from face_recog import encode_face, recognize_faces
from camera import start_camera, stop_camera, get_frame
from face_worker import worker
from face_utils import is_face_duplicate


# --- Constants ---
FRAME_QUEUE_MAX = 4
RESULT_QUEUE_MAX = 4
WORKER_RESTART_DELAY = 0.3

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Face Attendance System")
        self.root.geometry("900x720")

        # Init DB
        init_db()

        # Top dashboard
        db_frame = tk.Frame(root)
        db_frame.pack(pady=6)
        self.total_people_label = tk.Label(db_frame, text="Total Tamu Terdaftar: 0", font=("Arial", 12), fg="blue")
        self.total_people_label.grid(row=0, column=0, padx=10)
        self.total_guest_label = tk.Label(db_frame, text="Total Undangan Keseluruhan: 0", font=("Arial", 12), fg="green")
        self.total_guest_label.grid(row=0, column=1, padx=10)

        # Video display
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=6)

        # Input form
        form = tk.Frame(root)
        form.pack(pady=6)
        tk.Label(form, text="Nama Tamu:").grid(row=0, column=0, sticky="e")
        self.name_entry = tk.Entry(form, width=30)
        self.name_entry.grid(row=0, column=1, padx=6)
        tk.Label(form, text="Jumlah tamu:").grid(row=1, column=0, sticky="e")
        self.guest_entry = tk.Entry(form, width=10)
        self.guest_entry.insert(0, "1")
        self.guest_entry.grid(row=1, column=1, padx=6, sticky="w")
        tk.Button(root, text="📸 Capture & Save", bg="green", fg="white", font=("Arial", 11),
                  command=self.capture_face).pack(pady=8)

        # load known faces
        self.known_encodings, self.known_names, self.known_guests = load_registered_faces()

        # queues and worker process placeholders
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_MAX)
        self.result_queue = Queue(maxsize=RESULT_QUEUE_MAX)
        self.worker_proc = None

        # start camera thread (puts frames into frame_queue)
        start_camera(frame_queue=self.frame_queue, fps_target=60)

        # start worker
        self.start_worker()

        # GUI state
        self.last_frame = None
        self.last_result = None

        # update dashboard and loop
        self.update_dashboard()
        self.root.after(5, self.update_loop)

        # proper close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- worker lifecycle ----------------
    def start_worker(self):
        # ensure previous terminated
        if self.worker_proc is not None and self.worker_proc.is_alive():
            try:
                self.worker_proc.terminate()
                time.sleep(WORKER_RESTART_DELAY)
            except Exception:
                pass

        # prepare serializable known_encodings (list of lists)
        known_enc_serial = [enc.tolist() for enc in self.known_encodings]
        self.worker_proc = Process(target=worker, args=(
            self.frame_queue, self.result_queue, known_enc_serial, self.known_names, self.known_guests
        ), daemon=True)
        self.worker_proc.start()
        print("[MAIN] Worker started, PID:", getattr(self.worker_proc, "pid", None))

    def restart_worker_after_db_change(self):
        # reload known faces then restart worker
        self.known_encodings, self.known_names, self.known_guests = load_registered_faces()
        # stop old worker and start new one
        self.start_worker()
        # update dashboard
        self.update_dashboard()

    # ---------------- GUI actions ----------------
    def update_dashboard(self):
        total_people, total_guests = get_dashboard_stats()
        self.total_people_label.config(text=f"Total Tamu Terdaftar: {total_people}")
        self.total_guest_label.config(text=f"Total Undangan Keseluruhan: {total_guests}")

    def capture_face(self):
        # get latest frame (non-blocking)
        frame = None
        try:
            # prefer using last_frame cached if queue empty
            frame = self.frame_queue.get_nowait()
        except Exception:
            frame = get_frame()

        if frame is None:
            messagebox.showerror("Camera", "Tidak ada frame kamera saat ini. Pastikan kamera terhubung.")
            return

        name = self.name_entry.get().strip()
        count = self.guest_entry.get().strip()

        if not name or not count.isdigit():
            messagebox.showwarning("Invalid", "Isi nama & jumlah tamu valid!")
            return

        # --- STEP 1: Cek wajah sudah pernah terdaftar atau belum ---
        small = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs, names, guests = recognize_faces(
            rgb_small, self.known_encodings, self.known_names, self.known_guests
        )

        if len(names) > 0 and names[0] != "Unknown":
            # Wajah sudah terdaftar → blok simpan
            messagebox.showwarning(
                "Sudah Terdaftar",
                f"Wajah ini sudah terdaftar sebagai '{names[0]}' 🚫"
            )
            return

        # --- STEP 2: Encode wajah untuk disimpan (karena baru) ---
        encoding, face = encode_face(frame)
        if encoding is None:
            messagebox.showwarning("No Face", "Tidak ada wajah terdeteksi!")
            return

        # Simpan ke DB
        save_face_to_db(name, int(count), encoding)

        # Reload database memory
        self.root.after(500, self.reload_faces)
        self.root.after(500, self.update_dashboard)

        messagebox.showinfo("Success ✅", "Wajah berhasil disimpan!")

        self.name_entry.delete(0, tk.END)
        self.guest_entry.delete(0, tk.END)
        self.guest_entry.insert(0, "1")

    def reload_faces(self):
        """
        Reload data wajah dari DB dan restart worker agar model update.
        """
        try:
            # load encodings & names again from DB
            self.known_encodings, self.known_names, self.known_guests = load_registered_faces()

            print(f"[MAIN] Reloaded faces: {len(self.known_names)} registered")

            # restart worker so new encodings are used
            self.start_worker()
            self.update_dashboard()

        except Exception as e:
            print("reload_faces error:", e)

    # ---------------- main GUI loop ----------------
    def update_loop(self):
        # try get latest frame from queue without blocking
        frame = None
        try:
            frame = self.frame_queue.get_nowait()
            self.last_frame = frame
        except Exception:
            frame = self.last_frame

        # try get latest recognition result
        try:
            res = self.result_queue.get_nowait()
            self.last_result = res
        except Exception:
            res = None

        if frame is not None:
            # draw boxes from last_result (if any)
            display_frame = frame.copy()
            if self.last_result is not None:
                locs, names, guests = self.last_result
                for (top, right, bottom, left), name, g in zip(locs, names, guests):
                    # scale coords back to original frame size (small->orig)
                    # small = 0.4 scale used in worker
                    top = int(top / 0.4)
                    right = int(right / 0.4)
                    bottom = int(bottom / 0.4)
                    left = int(left / 0.4)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = f"{name} ({g})" if name != "Unknown" else "Unknown"
                    cv2.putText(display_frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # convert BGR->RGB->PIL->ImageTk
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            im_pil = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.config(image=im_pil)
            self.video_label.image = im_pil

        # schedule next loop
        self.root.after(8, self.update_loop)

    # ---------------- cleanup ----------------
    def on_close(self):
        try:
            if self.worker_proc is not None and self.worker_proc.is_alive():
                self.worker_proc.terminate()
        except Exception:
            pass
        try:
            stop_camera()
        except Exception:
            pass
        self.root.destroy()

#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = FaceApp(root)
#     root.mainloop()
