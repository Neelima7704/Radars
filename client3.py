# radar_client.py
import socket
import pickle
import pandas as pd
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time

# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 65432

WINDOW_SIZE = 2500
EPS = 0.8
MIN_SAMPLES = 20

DOA_GATE = 5.0
FREQ_GATE = 20.0
# --------------------------------------

# ---------------- STORAGE ----------------
pulse_buffer = deque(maxlen=WINDOW_SIZE)
emitters = {}
emitter_id_counter = 0
scaler = StandardScaler()

last_toa = None
# ----------------------------------------

# -------- EMITTER FUNCTIONS --------
def associate_emitter(stats):
    for eid, e in emitters.items():
        if abs(stats["meanDOA"] - e["meanDOA"]) < DOA_GATE and \
           abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
            return eid
    return None

def update_emitter(eid, stats):
    e = emitters[eid]
    e["meanDOA"]  = 0.7 * e["meanDOA"]  + 0.3 * stats["meanDOA"]
    e["meanFreq"] = 0.7 * e["meanFreq"] + 0.3 * stats["meanFreq"]
    e["minDOA"]   = min(e["minDOA"], stats["minDOA"])
    e["maxDOA"]   = max(e["maxDOA"], stats["maxDOA"])
    e["minFreq"]  = min(e["minFreq"], stats["minFreq"])
    e["maxFreq"]  = max(e["maxFreq"], stats["maxFreq"])
    e["pulseCount"] += stats["pulseCount"]

def create_emitter(stats):
    global emitter_id_counter
    emitters[emitter_id_counter] = {
        "meanDOA": stats["meanDOA"],
        "meanFreq": stats["meanFreq"],
        "minDOA": stats["minDOA"],
        "maxDOA": stats["maxDOA"],
        "minFreq": stats["minFreq"],
        "maxFreq": stats["maxFreq"],
        "pulseCount": stats["pulseCount"]
    }
    emitter_id_counter += 1

# -------- PROCESS WINDOW --------
def process_window(buffer):
    if len(buffer) < MIN_SAMPLES:
        return

    df = pd.DataFrame(buffer)

    X = scaler.fit_transform(
        df[["doa", "frequency", "pulse_width", "pri"]]
    )

    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)
    df["cluster"] = labels

    for cid in set(labels):
        if cid == -1:
            continue

        c = df[df["cluster"] == cid]
        stats = {
            "meanDOA": c["doa"].mean(),
            "meanFreq": c["frequency"].mean(),
            "minDOA": c["doa"].min(),
            "maxDOA": c["doa"].max(),
            "minFreq": c["frequency"].min(),
            "maxFreq": c["frequency"].max(),
            "pulseCount": len(c)
        }

        eid = associate_emitter(stats)
        if eid is None:
            create_emitter(stats)
        else:
            update_emitter(eid, stats)

# ---------------- MAIN CLIENT ----------------
if __name__ == "__main__":
    start_time = time.perf_counter()
    window_count = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("Connected to server")

        while True:
            length_bytes = client.recv(8)
            if not length_bytes:
                break

            data_length = int.from_bytes(length_bytes, "big")
            payload = b""

            while len(payload) < data_length:
                payload += client.recv(data_length - len(payload))

            raw_window = pickle.loads(payload)
            window_count += 1
            print(f"t{window_count} → pulses arrived = {len(raw_window)}")

            # --- TOA → PRI ---
            for row in raw_window:
                toa = row["toa"]

                if last_toa is None:
                    pri = 0.0
                else:
                    pri = toa - last_toa

                last_toa = toa

                pulse_buffer.append({
                    "doa": row["doa"],
                    "frequency": row["frequency"],
                    "pulse_width": row["pulse_width"],
                    "pri": pri
                })

            process_window(pulse_buffer)

    end_time = time.perf_counter()

    # ---------------- OUTPUT ----------------
    print("\nFINAL RADAR EMITTER SUMMARY")
    print(f"Total Emitters Detected: {len(emitters)}\n")

    for eid, e in emitters.items():
        print(
            f"Emitter {eid} | "
            f"DOA {e['meanDOA']:.2f}° "
            f"[{e['minDOA']:.2f}-{e['maxDOA']:.2f}] | "
            f"Freq {e['meanFreq']:.2f} MHz "
            f"[{e['minFreq']:.2f}-{e['maxFreq']:.2f}] | "
            f"Pulses {e['pulseCount']}"
        )

    print("\nTotal Processing Time:")
    print(f"{end_time - start_time:.3f} seconds")



