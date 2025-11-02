from database import init_db
from gui import FaceApp
import tkinter as tk
from camera import stop_camera

init_db()

root = tk.Tk()
app = FaceApp(root)

def on_close(self):
    stop_camera()
    self.root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
