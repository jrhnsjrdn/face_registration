from database import init_db
from gui import FaceApp
import tkinter as tk

init_db()

root = tk.Tk()
app = FaceApp(root)

root.mainloop()
