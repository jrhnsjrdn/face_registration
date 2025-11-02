import multiprocessing

from gui import FaceApp
import tkinter as tk

if __name__ == "__main__":
    # optional kalau nanti freeze pakai PyInstaller
    multiprocessing.freeze_support()

    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
