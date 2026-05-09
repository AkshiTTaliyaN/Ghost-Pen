import os
import sys
import platform
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import threading
import json

# ── Resolve config ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from config import DATA_DIR, LABELS_PATH, SCRIPT_DIR

PYTHON = sys.executable

COLLECT_SCRIPT   = os.path.join(SCRIPT_DIR, "collecting-data.py")
TRAIN_SCRIPT     = os.path.join(SCRIPT_DIR, "training-data.py")
RECOGNIZE_SCRIPT = os.path.join(SCRIPT_DIR, "recognizing-real-time.py")
CHECK_SCRIPT     = os.path.join(SCRIPT_DIR, "check.py")


def open_new_console(cmd: list[str], cwd: str, env: dict) -> subprocess.Popen:
    """
    Launch a subprocess in a new terminal window, cross-platform.
    - Windows : CREATE_NEW_CONSOLE
    - macOS   : open -a Terminal (via AppleScript)
    - Linux   : tries x-terminal-emulator, gnome-terminal, xterm in order
    """
    system = platform.system()

    if system == "Windows":
        return subprocess.Popen(
            cmd, cwd=cwd, env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    if system == "Darwin":
        # macOS: write a temp shell script and open it in Terminal
        import tempfile, stat
        script_content = (
            "#!/bin/bash\n"
            f"cd {cwd}\n"
            + " ".join(f'"{c}"' for c in cmd)
            + "\nread -p 'Press Enter to close...'\n"
        )
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.sh', delete=False
        )
        tmp.write(script_content)
        tmp.close()
        os.chmod(tmp.name, stat.S_IRWXU)
        return subprocess.Popen(
            ["open", "-a", "Terminal", tmp.name],
            cwd=cwd, env=env,
        )

    # Linux: try common terminal emulators
    terminals = [
        ["x-terminal-emulator", "-e"],
        ["gnome-terminal", "--"],
        ["xterm", "-e"],
        ["konsole", "-e"],
    ]
    for term_prefix in terminals:
        try:
            return subprocess.Popen(
                term_prefix + cmd, cwd=cwd, env=env,
            )
        except FileNotFoundError:
            continue

    # Fallback: run without new window (still works, just no visible console)
    return subprocess.Popen(cmd, cwd=cwd, env=env)


class GhostPenGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GhostPen — Air Drawing System")
        self.root.geometry("420x440")
        self.root.resizable(False, False)

        tk.Label(
            root,
            text="GhostPen",
            font=("Helvetica", 20, "bold"),
        ).pack(pady=(20, 2))

        tk.Label(
            root,
            text="Air Drawing Gesture Recognition",
            font=("Helvetica", 10),
            fg="gray",
        ).pack(pady=(0, 16))

        self.add_button("Collect Gesture Data",    self.collect_data)
        self.add_button("Train Model",             self.train_model)
        self.add_button("Start Live Recognition",  self.start_recognition)
        self.add_button("Dataset Stats",           self.show_stats)
        self.add_button("Exit",                    self.exit_app)

        # Log panel
        self.log = scrolledtext.ScrolledText(
            root, height=6, font=("Courier", 9), state='disabled',
            bg="#1e1e1e", fg="#d4d4d4",
        )
        self.log.pack(fill='x', padx=12, pady=(10, 12))

        self.recognition_process: subprocess.Popen | None = None
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

    def add_button(self, text: str, command):
        tk.Button(
            self.root, text=text,
            font=("Arial", 11), width=32,
            command=command,
        ).pack(pady=4)

    def log_line(self, msg: str):
        self.log.configure(state='normal')
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state='disabled')

    # ── Collect data ──────────────────────────────────────────────────────────
    def collect_data(self):
        label = simpledialog.askstring("Gesture Label", "Enter gesture label (e.g. A, B, 0):")
        if not label or not label.strip():
            messagebox.showerror("Error", "Gesture label cannot be empty.")
            return

        label = label.strip().upper()
        env   = {**os.environ, "GESTURE_NAME": label}

        try:
            subprocess.run(
                [PYTHON, COLLECT_SCRIPT],
                cwd=BASE_DIR, env=env, check=True,
            )
            self.log_line(f"Collection complete for '{label}'.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Data Collection Failed", str(e))
            self.log_line(f"ERROR collecting '{label}': {e}")

    # ── Train model ───────────────────────────────────────────────────────────
    def train_model(self):
        self.log_line("Starting training — this may take a while...")

        def run():
            try:
                result = subprocess.run(
                    [PYTHON, TRAIN_SCRIPT],
                    cwd=BASE_DIR, check=True,
                    capture_output=True, text=True,
                )
                self.root.after(0, lambda: self.log_line(
                    "Training complete.\n" + result.stdout[-800:]
                ))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Training Done", "Model trained and saved successfully."
                ))
            except subprocess.CalledProcessError as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Training Failed", e.stderr[-400:] if e.stderr else str(e)
                ))

        threading.Thread(target=run, daemon=True).start()

    # ── Live recognition ──────────────────────────────────────────────────────
    def start_recognition(self):
        if self.recognition_process and self.recognition_process.poll() is None:
            messagebox.showinfo("Running", "Live recognition is already running.")
            return

        try:
            self.recognition_process = open_new_console(
                cmd=[PYTHON, RECOGNIZE_SCRIPT],
                cwd=BASE_DIR,
                env=os.environ.copy(),
            )
            self.log_line("Live recognition started.")
            messagebox.showinfo(
                "Started",
                "Live recognition launched.\n\n"
                "• Press 'Q' inside the camera window to stop\n"
                "• Backspace = delete last letter\n"
                "• Space key = insert space\n"
                "• 'C' key = clear word buffer",
            )
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))
            self.log_line(f"ERROR: {e}")

    # ── Dataset stats ─────────────────────────────────────────────────────────
    def show_stats(self):
        if not os.path.isdir(DATA_DIR):
            messagebox.showinfo("No Data", f"Data directory not found:\n{DATA_DIR}")
            return

        lines = ["Sample counts per class:\n"]
        total = 0
        for label in sorted(os.listdir(DATA_DIR)):
            folder = os.path.join(DATA_DIR, label)
            if not os.path.isdir(folder):
                continue
            count = len([f for f in os.listdir(folder) if f.endswith(".npy")])
            flag  = "  ✓" if count >= 500 else f"  ← need {500 - count} more"
            lines.append(f"  {label:>4}: {count:>5} samples{flag}")
            total += count

        lines.append(f"\n  TOTAL: {total} sequences")
        messagebox.showinfo("Dataset Stats", "\n".join(lines))
        self.log_line(f"Stats: {total} total sequences across {len(lines)-2} classes.")

    # ── Exit ──────────────────────────────────────────────────────────────────
    def exit_app(self):
        if self.recognition_process and self.recognition_process.poll() is None:
            self.recognition_process.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = GhostPenGUI(root)
    root.mainloop()
