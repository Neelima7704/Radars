import socket
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
import threading


# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 65432

WINDOW_SIZE = 10000
EPS = 0.9
MIN_SAMPLES = 20

DOA_GATE = 5.0
FREQ_GATE = 20.0

SHIP_DOA_GATE = 5.0
SHIP_FREQ_SPAN = 200.0


# ---------------- STORAGE ----------------
pulse_buffer = []
emitters = {}
emitter_id_counter = 0
scaler = StandardScaler()
last_toa = None

data_finished = False
window_ready = False
lock = threading.Lock()
window_count = 0


# -------- EMITTER FUNCTIONS --------
def associate_emitter(stats):
    for eid, e in emitters.items():
        if abs(stats["meanDOA"] - e["meanDOA"]) < DOA_GATE and \
           abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
            return eid
    return None


def update_emitter(eid, stats):
    e = emitters[eid]
    e["meanDOA"] = 0.7 * e["meanDOA"] + 0.3 * stats["meanDOA"]
    e["meanFreq"] = 0.7 * e["meanFreq"] + 0.3 * stats["meanFreq"]
    e["minDOA"] = min(e["minDOA"], stats["minDOA"])
    e["maxDOA"] = max(e["maxDOA"], stats["maxDOA"])
    e["minFreq"] = min(e["minFreq"], stats["minFreq"])
    e["maxFreq"] = max(e["maxFreq"], stats["maxFreq"])
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


# -------- SHIP GROUPING --------
def group_emitters_into_ships(emitters):
    ships = []

    for eid, e in emitters.items():
        assigned = False

        for ship in ships:
            doa_close = abs(e["meanDOA"] - ship["meanDOA"]) < SHIP_DOA_GATE
            combined_min_freq = min(ship["minFreq"], e["minFreq"])
            combined_max_freq = max(ship["maxFreq"], e["maxFreq"])
            freq_span_ok = (combined_max_freq - combined_min_freq) < SHIP_FREQ_SPAN

            if doa_close and freq_span_ok:
                ship["meanDOA"] = 0.7 * ship["meanDOA"] + 0.3 * e["meanDOA"]
                ship["minFreq"] = combined_min_freq
                ship["maxFreq"] = combined_max_freq
                ship["emitters"].append(eid)
                assigned = True
                break

        if not assigned:
            ships.append({
                "meanDOA": e["meanDOA"],
                "minFreq": e["minFreq"],
                "maxFreq": e["maxFreq"],
                "emitters": [eid]
            })

    return ships


# -------- PROCESS WINDOW --------
def process_window(local_buffer):
    if len(local_buffer) < MIN_SAMPLES:
        return

    df = pd.DataFrame(local_buffer)

    X = scaler.fit_transform(df[["doa", "frequency", "pulse_width", "pri"]])
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


# ---------------- THREADS ----------------

def receive_data(client):
    global last_toa, window_count, data_finished, window_ready

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

        with lock:
            for row in raw_window:
                toa = row["toa"]
                pri = 0.0 if last_toa is None else toa - last_toa
                last_toa = toa

                pulse_buffer.append({
                    "doa": row["doa"],
                    "frequency": row["frequency"],
                    "pulse_width": row["pulse_width"],
                    "pri": pri
                })

            window_ready = True

    data_finished = True


def processing_thread():
    global window_ready

    while not data_finished or window_ready:

        if window_ready:
            with lock:
                local_copy = pulse_buffer.copy()
                pulse_buffer.clear()
                window_ready = False

            process_window(local_copy)

        time.sleep(0.01)


# ---------------- MAIN ----------------
if __name__ == "__main__":

    start_time = time.perf_counter()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("Connected to server")

        t1 = threading.Thread(target=receive_data, args=(client,))
        t2 = threading.Thread(target=processing_thread)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

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

    ships = group_emitters_into_ships(emitters)

    print("\n--- SHIP LEVEL SUMMARY ---")
    print(f"Total number of emitters formed = {len(emitters)}")
    print(f"Possible number of ships = {len(ships)}")

    print("\nTotal Processing Time:")
    print(f"{end_time - start_time:.3f} seconds")
