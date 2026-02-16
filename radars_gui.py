import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# ---------------- CONFIG ----------------
EPS = 0.9
MIN_SAMPLES = 20
DOA_GATE = 5.0
FREQ_GATE = 20.0


# ---------------- PROCESSING FUNCTION ----------------
def process_directory(folder_path, output_box):

    start_time = time.perf_counter()

    emitters = {}
    emitter_id_counter = 0
    scaler = StandardScaler()
    processed_pulses = set()

    all_pulses = []

    # -------- READ CSV FILES --------
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            full_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(full_path)

                for _, row in df.iterrows():

                    pulse_id = (
                        row["toa"],
                        row["doa"],
                        row["frequency"],
                        row["pulse_width"],
                    )

                    if pulse_id in processed_pulses:
                        continue

                    processed_pulses.add(pulse_id)

                    pulse = {
                        "doa": row["doa"],
                        "frequency": row["frequency"],
                        "pulse_width": row["pulse_width"],
                        "pri": 0  # simple placeholder
                    }

                    all_pulses.append(pulse)

            except Exception as e:
                output_box.insert(tk.END, f"Error reading {file}: {e}\n")

    if len(all_pulses) < MIN_SAMPLES:
        output_box.insert(tk.END, "Not enough pulses for clustering.\n")
        return

    # -------- CLUSTERING --------
    df_all = pd.DataFrame(all_pulses)
    X = scaler.fit_transform(df_all[["doa", "frequency", "pulse_width", "pri"]])
    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)
    df_all["cluster"] = labels

    for cid in set(labels):
        if cid == -1:
            continue

        c = df_all[df_all["cluster"] == cid]

        stats = {
            "meanDOA": c["doa"].mean(),
            "meanFreq": c["frequency"].mean(),
            "minDOA": c["doa"].min(),
            "maxDOA": c["doa"].max(),
            "minFreq": c["frequency"].min(),
            "maxFreq": c["frequency"].max(),
            "pulseCount": len(c)
        }

        emitters[emitter_id_counter] = stats
        emitter_id_counter += 1

    end_time = time.perf_counter()

    # -------- OUTPUT --------
    output_box.delete("1.0", tk.END)

    output_box.insert(tk.END, "FINAL RADAR EMITTER SUMMARY\n")
    output_box.insert(tk.END, f"Total Emitters Detected: {len(emitters)}\n\n")

    for eid, e in emitters.items():
        output_box.insert(
            tk.END,
            f"Emitter {eid} | "
            f"DOA {e['meanDOA']:.2f}° "
            f"[{e['minDOA']:.2f}-{e['maxDOA']:.2f}] | "
            f"Freq {e['meanFreq']:.2f} MHz "
            f"[{e['minFreq']:.2f}-{e['maxFreq']:.2f}] | "
            f"Pulses {e['pulseCount']}\n"
        )

    output_box.insert(tk.END, "\nTotal Processing Time:\n")
    output_box.insert(tk.END, f"{end_time - start_time:.3f} seconds\n")


# ---------------- GUI FUNCTIONS ----------------
def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        folder_path_var.set(folder_selected)


def start_processing():
    folder = folder_path_var.get()

    if not folder:
        messagebox.showwarning("Warning", "Please select a folder first!")
        return

    process_directory(folder, output_box)


# ---------------- GUI WINDOW ----------------
root = tk.Tk()
root.title("Radar Emitter Processing Tool")
root.geometry("750x500")
root.resizable(False, False)

folder_path_var = tk.StringVar()

# Title
tk.Label(root,
         text="Radar CSV Emitter Analyzer",
         font=("Arial", 16, "bold")).pack(pady=10)

# Folder Selection
frame = tk.Frame(root)
frame.pack(pady=5)

tk.Entry(frame,
         textvariable=folder_path_var,
         width=60).pack(side=tk.LEFT, padx=5)

tk.Button(frame,
          text="Browse",
          command=browse_folder).pack(side=tk.LEFT)

# Start Button
tk.Button(root,
          text="Start Processing",
          font=("Arial", 12),
          bg="green",
          fg="white",
          command=start_processing).pack(pady=10)

# Output Box
output_box = scrolledtext.ScrolledText(root,
                                       width=90,
                                       height=18,
                                       font=("Consolas", 10))
output_box.pack(pady=10)

root.mainloop()
