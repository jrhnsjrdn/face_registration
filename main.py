from database import init_db
from gui import FaceApp
import tkinter as tk
from camera import release_cam

init_db()

root = tk.Tk()
app = FaceApp(root)

def on_close():
    release_cam()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
