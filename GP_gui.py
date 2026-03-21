import os
import sys
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox

# -------------------------------------------------
# Python interpreter (same env)
# -------------------------------------------------
PYTHON = sys.executable

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(BASE_DIR, "script")

COLLECT_SCRIPT = os.path.join(SCRIPT_DIR, "collecting-data.py")
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "training-data.py")
RECOGNIZE_SCRIPT = os.path.join(SCRIPT_DIR, "recognizing-real-time.py")

# -------------------------------------------------
# GUI Class
# -------------------------------------------------
class GhostPenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GhostPen - Air Drawing System")
        self.root.geometry("360x320")
        self.root.resizable(False, False)

        tk.Label(
            root,
            text="GhostPen - Air Drawing System",
            font=("Helvetica", 16, "bold")
        ).pack(pady=20)

        self.add_button("Collect Gesture Data", self.collect_data)
        self.add_button("Train Model", self.train_model)
        self.add_button("Start Live Recognition", self.start_recognition)
        self.add_button("Exit", self.exit_app)

        self.recognition_process = None
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

    def add_button(self, text, command):
        tk.Button(
            self.root,
            text=text,
            font=("Arial", 12),
            width=30,
            command=command
        ).pack(pady=8)

    # -------------------------------------------------
    # Collect data (UNCHANGED)
    # -------------------------------------------------
    def collect_data(self):
        label = simpledialog.askstring("Gesture Label", "Enter gesture label:")
        if not label or not label.strip():
            messagebox.showerror("Error", "Gesture label cannot be empty.")
            return

        try:
            subprocess.run(
                [PYTHON, COLLECT_SCRIPT],
                cwd=BASE_DIR,
                env={**os.environ, "GESTURE_NAME": label.strip()},
                check=True
            )
            messagebox.showinfo("Success", f"Gesture '{label}' collected.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Data Collection Failed", str(e))

    # -------------------------------------------------
    # Train model (UNCHANGED)
    # -------------------------------------------------
    def train_model(self):
        try:
            subprocess.run(
                [PYTHON, TRAIN_SCRIPT],
                cwd=BASE_DIR,
                check=True
            )
            messagebox.showinfo("Training Done", "Model trained successfully.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Training Failed", str(e))

    # -------------------------------------------------
    # Start Live Recognition (FIXED ONLY HERE)
    # -------------------------------------------------
    def start_recognition(self):
        if self.recognition_process and self.recognition_process.poll() is None:
            messagebox.showinfo("Running", "Live recognition is already running.")
            return

        try:
            self.recognition_process = subprocess.Popen(
                [PYTHON, RECOGNIZE_SCRIPT],
                cwd=BASE_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE  # 🔥 THIS IS THE FIX
            )

            messagebox.showinfo(
                "Started",
                "Live recognition started.\n\n"
                "• Camera window should open\n"
                "• Press 'Q' to stop recognition"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recognition:\n{e}")

    def exit_app(self):
        if self.recognition_process and self.recognition_process.poll() is None:
            self.recognition_process.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GhostPenGUI(root)
    root.mainloop()
