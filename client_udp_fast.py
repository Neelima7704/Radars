import socket
import pickle
import pandas as pd
import numpy as np
import threading
import queue
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 65432

EPS = 0.8
MIN_SAMPLES = 20

DOA_GATE = 5
FREQ_GATE = 20

SHIP_DOA_GATE = 5
SHIP_FREQ_SPAN = 200
# ----------------------------------------

packet_queue = queue.Queue()
emitters = {}
emitter_id_counter = 0
last_toa = None
window_counter = 0   # ✅ Added for printing window count


# ---------------- UTIL ----------------
def angular_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


# ---------------- EMITTER ASSOCIATION ----------------
def associate_emitter(stats):
    for eid, e in emitters.items():
        if angular_distance(stats["meanDOA"], e["meanDOA"]) < DOA_GATE and \
           abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
            return eid
    return None


def update_emitter(eid, stats):
    e = emitters[eid]

    e["meanDOA"] = (0.7*e["meanDOA"] + 0.3*stats["meanDOA"]) % 360
    e["meanFreq"] = 0.7*e["meanFreq"] + 0.3*stats["meanFreq"]

    e["minDOA"] = min(e["minDOA"], stats["minDOA"])
    e["maxDOA"] = max(e["maxDOA"], stats["maxDOA"])
    e["minFreq"] = min(e["minFreq"], stats["minFreq"])
    e["maxFreq"] = max(e["maxFreq"], stats["maxFreq"])

    e["pulseCount"] += stats["pulseCount"]


def create_emitter(stats):
    global emitter_id_counter
    emitters[emitter_id_counter] = stats
    emitter_id_counter += 1


# ---------------- SHIP GROUPING ----------------
def group_emitters_into_ships():

    ships = []

    for eid, e in emitters.items():
        assigned = False

        for ship in ships:
            doa_close = angular_distance(e["meanDOA"], ship["meanDOA"]) < SHIP_DOA_GATE

            combined_min = min(ship["minFreq"], e["minFreq"])
            combined_max = max(ship["maxFreq"], e["maxFreq"])
            freq_ok = (combined_max - combined_min) < SHIP_FREQ_SPAN

            if doa_close and freq_ok:
                ship["meanDOA"] = (0.7*ship["meanDOA"] + 0.3*e["meanDOA"]) % 360
                ship["minFreq"] = combined_min
                ship["maxFreq"] = combined_max
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

    # --------- Weighted Mean Frequency ----------
    for ship in ships:
        total_freq = 0
        total_pulses = 0

        for eid in ship["emitters"]:
            total_freq += emitters[eid]["meanFreq"] * emitters[eid]["pulseCount"]
            total_pulses += emitters[eid]["pulseCount"]

        ship["meanFreq"] = total_freq / total_pulses if total_pulses > 0 else 0

    return ships


# ---------------- PROCESS WINDOW ----------------
def process_window(window_data):

    if len(window_data) < MIN_SAMPLES:
        return

    df = pd.DataFrame(window_data)

    df["doa_sin"] = np.sin(np.deg2rad(df["doa"]))
    df["doa_cos"] = np.cos(np.deg2rad(df["doa"]))

    X = StandardScaler().fit_transform(
        df[["doa_sin", "doa_cos", "frequency", "pulse_width", "pri"]].values
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


# ---------------- RECEIVER THREAD ----------------
def receiver(sock):

    buffers = {}

    while True:
        data, _ = sock.recvfrom(65536)

        if data == b"END":
            packet_queue.put("END")
            break

        header, chunk = data.split(b"|", 1)
        cid, total = header.decode().split("/")
        cid = int(cid)
        total = int(total)

        if "cur" not in buffers:
            buffers["cur"] = [None]*total
            buffers["count"] = 0

        if buffers["cur"][cid] is None:
            buffers["count"] += 1

        buffers["cur"][cid] = chunk

        if buffers["count"] == total:
            payload = b"".join(buffers["cur"])
            packet_queue.put(payload)
            buffers.clear()


# ---------------- PROCESSOR THREAD ----------------
def processor():

    global last_toa, window_counter

    while True:

        payload = packet_queue.get()

        if payload == "END":
            break

        raw_window = pickle.loads(payload)

        window_counter += 1
        print(f"t{window_counter} → received {len(raw_window)} pulses")

        window_data = []

        for row in raw_window:
            toa = row["toa"]
            pri = 0 if last_toa is None else toa - last_toa
            last_toa = toa

            window_data.append({
                "doa": row["doa"],
                "frequency": row["frequency"],
                "pulse_width": row["pulse_width"],
                "pri": pri
            })

        process_window(window_data)


# ---------------- MAIN ----------------
if __name__ == "__main__":

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10*1024*1024)

    client.sendto(b"Hello", (HOST, PORT))
    print("Connected to server\n")

    t1 = threading.Thread(target=receiver, args=(client,))
    t2 = threading.Thread(target=processor)

    start_time = time.perf_counter()

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    end_time = time.perf_counter()

    # --------------- FINAL REPORT ----------------

    print("\nRADAR EMITTER ANALYSIS REPORT".center(65))
    print(f"\nTotal Emitters Detected : {len(emitters)}\n")

    for eid, e in emitters.items():
        print(
        f"Emitter {eid:<2} | "
        f"DOA {e['meanDOA']:7.2f}° "
        f"[{e['minDOA']:7.2f}-{e['maxDOA']:7.2f}]° | "
        f"Freq {e['meanFreq']:8.2f} MHz "
        f"[{e['minFreq']:7.2f}-{e['maxFreq']:7.2f}] MHz | "
        f"Pulses {e['pulseCount']}"
    )


    ships = group_emitters_into_ships()

    print("\nSHIP LEVEL SUMMARY\n")
    print(f"Total Ships Detected : {len(ships)}\n")

    for i, ship in enumerate(ships):
        print(
            f"Ship {i:<2} | "
            f"DOA ~ {ship['meanDOA']:7.2f}° | "
            f"Mean Freq {ship['meanFreq']:8.2f} MHz | "
            f"Freq Span [{ship['minFreq']:7.2f}-{ship['maxFreq']:7.2f}] MHz | "
            f"Emitters {ship['emitters']}"
        )

    print("\nTotal Processing Time :", round(end_time - start_time, 3), "seconds")
