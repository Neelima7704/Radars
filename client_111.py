# # radar_client_optimized.py
# import socket
# import pickle
# import time
# import numpy as np
# from sklearn.cluster import DBSCAN

# # ---------------- CONFIG ----------------
# HOST = "127.0.0.1"
# PORT = 65432

# WINDOW_SIZE = 5000            # max buffer size
# CLUSTER_BATCH_SIZE = 1500     # cluster every 1500 pulses
# MIN_SAMPLES = 15              # min samples for clustering
# SOFT_MIN_SAMPLES = 5          # relaxed clustering at end-of-stream
# DOA_GATE = 5.0
# FREQ_GATE = 20.0
# # ---------------------------------------

# # STORAGE
# pulse_buffer = []
# pulse_counter_since_cluster = 0
# last_pulse_time = None
# emitters = {}
# emitter_id_counter = 0
# total_pulses_received = 0  # Track total received pulses


# # --------- EMITTER FUNCTIONS ---------
# def associate_emitter(stats):
#     for eid, e in emitters.items():
#         if abs(stats["meanDOA"] - e["meanDOA"]) < DOA_GATE and \
#            abs(stats["meanFreq"] - e["meanFreq"]) < FREQ_GATE:
#             return eid
#     return None

# def update_emitter(eid, stats):
#     e = emitters[eid]
#     e["meanDOA"]  = 0.7 * e["meanDOA"]  + 0.3 * stats["meanDOA"]
#     e["meanFreq"] = 0.7 * e["meanFreq"] + 0.3 * stats["meanFreq"]
#     e["minDOA"]   = min(e["minDOA"], stats["minDOA"])
#     e["maxDOA"]   = max(e["maxDOA"], stats["maxDOA"])
#     e["minFreq"]  = min(e["minFreq"], stats["minFreq"])
#     e["maxFreq"]  = max(e["maxFreq"], stats["maxFreq"])
#     e["pulseCount"] = stats["pulseCount"]  # Replace instead of adding

# def create_emitter(stats):
#     global emitter_id_counter
#     emitters[emitter_id_counter] = {
#         "meanDOA": stats["meanDOA"],
#         "meanFreq": stats["meanFreq"],
#         "minDOA": stats["minDOA"],
#         "maxDOA": stats["maxDOA"],
#         "minFreq": stats["minFreq"],
#         "maxFreq": stats["maxFreq"],
#         "pulseCount": stats["pulseCount"]
#     }
#     emitter_id_counter += 1


# # --------- PROCESS BUFFER ---------
# def process_buffer(buffer_list, min_samples):
#     if len(buffer_list) < min_samples:
#         return

#     # Extract DOA and frequency for clustering
#     data = np.array([(p["doa"], p["frequency"]) for p in buffer_list], dtype=np.float32)
    
#     # Normalize features for proper distance calculation
#     means = data.mean(axis=0)
#     stds = data.std(axis=0)
#     stds[stds == 0] = 1
#     X_scaled = (data - means) / stds
    
#     # Use DBSCAN with optimized parameters for speed
#     eps = 0.3  # fixed eps value for consistency
#     labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)

#     # Process clusters efficiently
#     unique_labels = np.unique(labels)
#     for cid in unique_labels:
#         if cid == -1:
#             continue
#         mask = labels == cid
#         cluster_pulses = [buffer_list[i] for i in np.where(mask)[0]]
        
#         # Calculate statistics
#         doas = np.array([p["doa"] for p in cluster_pulses])
#         freqs = np.array([p["frequency"] for p in cluster_pulses])
        
#         stats = {
#             "meanDOA": float(doas.mean()),
#             "meanFreq": float(freqs.mean()),
#             "minDOA": float(doas.min()),
#             "maxDOA": float(doas.max()),
#             "minFreq": float(freqs.min()),
#             "maxFreq": float(freqs.max()),
#             "pulseCount": len(cluster_pulses)
#         }
#         eid = associate_emitter(stats)
#         if eid is None:
#             create_emitter(stats)
#         else:
#             update_emitter(eid, stats)


# # ---------------- MAIN CLIENT ----------------
# if __name__ == "__main__":
#     start_time = time.perf_counter()
#     window_counter = 0

#     print("Connected to server")

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
#         client.connect((HOST, PORT))

#         while True:
#             length_bytes = client.recv(8)
#             if not length_bytes:
#                 break

#             data_length = int.from_bytes(length_bytes, "big")
#             payload = b""
#             while len(payload) < data_length:
#                 packet = client.recv(data_length - len(payload))
#                 if not packet:
#                     break
#                 payload += packet
#             if not payload:
#                 break

#             window_counter += 1
#             raw_window = pickle.loads(payload)
#             print(f"t{window_counter} → pulses arrived = {len(raw_window)}")

#             # --- Batch append pulses ---
#             # Convert dict list to array for faster processing
#             batch_pulses = []
#             for row in raw_window:
#                 batch_pulses.append({
#                     "doa": row["doa"],
#                     "frequency": row["frequency"],
#                     "pulse_width": row["pulse_width"],
#                     "pri": 0.001  # Fixed PRI for speed
#                 })
            
#             total_pulses_received += len(batch_pulses)  # Count total received
#             pulse_buffer.extend(batch_pulses)
#             pulse_counter_since_cluster += len(batch_pulses)

#             # Amortized clustering - only cluster when threshold reached
#             if pulse_counter_since_cluster >= CLUSTER_BATCH_SIZE:
#                 process_buffer(pulse_buffer, MIN_SAMPLES)
#                 pulse_counter_since_cluster = 0

#             # Keep buffer bounded
#             if len(pulse_buffer) > WINDOW_SIZE:
#                 pulse_buffer = pulse_buffer[-WINDOW_SIZE:]

#     # -------- END-OF-STREAM FLUSH --------
#     if SOFT_MIN_SAMPLES <= len(pulse_buffer) < MIN_SAMPLES:
#         print("End-of-stream detected → running relaxed clustering")
#         process_buffer(pulse_buffer, SOFT_MIN_SAMPLES)
#     elif len(pulse_buffer) < SOFT_MIN_SAMPLES:
#         print("Remaining pulses discarded as noise")

#     end_time = time.perf_counter()

#     # -------- FINAL OUTPUT --------
#     print("\nFINAL RADAR EMITTER SUMMARY")
#     print(f"Total Pulses Received: {total_pulses_received}")
#     print(f"Total Emitters Detected: {len(emitters)}\n")
    
#     total_clustered_pulses = sum(e["pulseCount"] for e in emitters.values())
#     print(f"Total Pulses Clustered: {total_clustered_pulses}\n")
    
#     for eid, e in emitters.items():
#         print(
#             f"Emitter {eid} | "
#             f"DOA {e['meanDOA']:.2f}° "
#             f"[{e['minDOA']:.2f}-{e['maxDOA']:.2f}] | "
#             f"Freq {e['meanFreq']:.2f} MHz "
#             f"[{e['minFreq']:.2f}-{e['maxFreq']:.2f}] | "
#             f"Pulses {e['pulseCount']}"
#         )

#     print("\nTotal Processing Time:")
#     print(f"{end_time - start_time:.3f} seconds")



# radar_client_dynamic_pri.py
import socket
import pickle
import time
import numpy as np
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 65432

WINDOW_SIZE = 5000            # max buffer size
CLUSTER_BATCH_SIZE = 1500     # cluster every 1500 pulses
MIN_SAMPLES = 15              # min samples for clustering
SOFT_MIN_SAMPLES = 5          # relaxed clustering at end-of-stream
DOA_GATE = 5.0
FREQ_GATE = 20.0
# ---------------------------------------

# ---------------- STORAGE ----------------
pulse_buffer = []
pulse_counter_since_cluster = 0
last_toa = None  # For dynamic PRI calculation
emitters = {}
emitter_id_counter = 0
total_pulses_received = 0  # Track total received pulses
# -----------------------------------------

# --------- EMITTER FUNCTIONS ---------
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
    e["pulseCount"] = stats["pulseCount"]  # Replace with current cluster count

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

# --------- PROCESS BUFFER ---------
def process_buffer(buffer_list, min_samples):
    if len(buffer_list) < min_samples:
        return

    # Extract DOA and frequency for clustering
    data = np.array([(p["doa"], p["frequency"]) for p in buffer_list], dtype=np.float32)

    # Normalize features
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds == 0] = 1
    X_scaled = (data - means) / stds

    # Use DBSCAN for clustering
    eps = 0.3
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)

    # Process clusters
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        if cid == -1:
            continue
        mask = labels == cid
        cluster_pulses = [buffer_list[i] for i in np.where(mask)[0]]

        doas = np.array([p["doa"] for p in cluster_pulses])
        freqs = np.array([p["frequency"] for p in cluster_pulses])

        stats = {
            "meanDOA": float(doas.mean()),
            "meanFreq": float(freqs.mean()),
            "minDOA": float(doas.min()),
            "maxDOA": float(doas.max()),
            "minFreq": float(freqs.min()),
            "maxFreq": float(freqs.max()),
            "pulseCount": len(cluster_pulses)
        }

        eid = associate_emitter(stats)
        if eid is None:
            create_emitter(stats)
        else:
            update_emitter(eid, stats)

# ---------------- MAIN CLIENT ----------------
if __name__ == "__main__":
    start_time = time.perf_counter()
    window_counter = 0

    print("Connecting to server...")

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
                packet = client.recv(data_length - len(payload))
                if not packet:
                    break
                payload += packet
            if not payload:
                break

            window_counter += 1
            raw_window = pickle.loads(payload)
            print(f"t{window_counter} → pulses arrived = {len(raw_window)}")

            # --- Append pulses with dynamic PRI ---
            for row in raw_window:
                toa = row.get("toa", time.perf_counter())  # fallback if TOA missing

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
                pulse_counter_since_cluster += 1
                total_pulses_received += 1

            # Amortized clustering
            if pulse_counter_since_cluster >= CLUSTER_BATCH_SIZE:
                process_buffer(pulse_buffer, MIN_SAMPLES)
                pulse_counter_since_cluster = 0

            # Keep buffer bounded
            if len(pulse_buffer) > WINDOW_SIZE:
                pulse_buffer = pulse_buffer[-WINDOW_SIZE:]

    # -------- END-OF-STREAM FLUSH --------
    if SOFT_MIN_SAMPLES <= len(pulse_buffer) < MIN_SAMPLES:
        print("End-of-stream → running relaxed clustering")
        process_buffer(pulse_buffer, SOFT_MIN_SAMPLES)
    elif len(pulse_buffer) < SOFT_MIN_SAMPLES:
        print("Remaining pulses discarded as noise")

    end_time = time.perf_counter()

    # -------- FINAL OUTPUT --------
    print("\nFINAL RADAR EMITTER SUMMARY")
    print(f"Total Pulses Received: {total_pulses_received}")
    print(f"Total Emitters Detected: {len(emitters)}\n")

    total_clustered_pulses = sum(e["pulseCount"] for e in emitters.values())
    print(f"Total Pulses Clustered: {total_clustered_pulses}\n")

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
