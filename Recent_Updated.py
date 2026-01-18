import time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler




WINDOW_SIZE = 1500
DBSCAN_EPS = 0.85
DBSCAN_MIN_SAMPLES = 10

DOA_GATE_DEG  = 4.0
FREQ_GATE_MHZ = 15.0
PRI_GATE_US   = 200.0

EWMA_ALPHA = 0.3


emitters = {}
next_emitter_id = 0
scaler = StandardScaler()


def find_emitter(stats):
    for eid, e in emitters.items():
        if (
            abs(stats["mean_doa"]  - e["mean_doa"])  < DOA_GATE_DEG and
            abs(stats["mean_freq"] - e["mean_freq"]) < FREQ_GATE_MHZ and
            abs(stats["mean_pri"]  - e["mean_pri"])  < PRI_GATE_US
        ):
            return eid
    return None



def update_emitter(eid, s):
    e = emitters[eid]

    e["mean_doa"]  = (1 - EWMA_ALPHA) * e["mean_doa"]  + EWMA_ALPHA * s["mean_doa"]
    e["mean_freq"] = (1 - EWMA_ALPHA) * e["mean_freq"] + EWMA_ALPHA * s["mean_freq"]
    e["mean_pri"]  = (1 - EWMA_ALPHA) * e["mean_pri"]  + EWMA_ALPHA * s["mean_pri"]

    e["min_doa"]  = min(e["min_doa"],  s["min_doa"])
    e["max_doa"]  = max(e["max_doa"],  s["max_doa"])
    e["min_freq"] = min(e["min_freq"], s["min_freq"])
    e["max_freq"] = max(e["max_freq"], s["max_freq"])

    e["pulse_widths"].extend(s["pulse_widths"])
    e["pulse_count"] += s["pulse_count"]

def create_emitter(s):
    global next_emitter_id
    emitters[next_emitter_id] = {
        "mean_doa": s["mean_doa"],
        "mean_freq": s["mean_freq"],
        "mean_pri": s["mean_pri"],
        "min_doa": s["min_doa"],
        "max_doa": s["max_doa"],
        "min_freq": s["min_freq"],
        "max_freq": s["max_freq"],
        "pulse_widths": list(s["pulse_widths"]),
        "pulse_count": s["pulse_count"]
    }
    next_emitter_id += 1



def process_window(df_window):
    if len(df_window) < DBSCAN_MIN_SAMPLES:
        return

    features = df_window[["doa", "frequency", "pulse_width", "pri"]]
    scaled_features = scaler.fit_transform(features)

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        n_jobs=-1
    ).fit_predict(scaled_features)

    df_window = df_window.copy()
    df_window["cluster"] = labels

    for cid in np.unique(labels):
        if cid == -1:
            continue

        cluster = df_window[df_window.cluster == cid]

        stats = {
            "mean_doa": cluster["doa"].mean(),
            "mean_freq": cluster["frequency"].mean(),
            "mean_pri": cluster["pri"].mean(),
            "min_doa": cluster["doa"].min(),
            "max_doa": cluster["doa"].max(),
            "min_freq": cluster["frequency"].min(),
            "max_freq": cluster["frequency"].max(),
            "pulse_widths": cluster["pulse_width"].values,
            "pulse_count": len(cluster)
        }

        eid = find_emitter(stats)
        if eid is None:
            create_emitter(stats)
        else:
            update_emitter(eid, stats)



if __name__ == "__main__":

    # CSV READ TIMING 
    
    csv_start = time.time()
    radar_df = pd.read_csv("DF2.csv")
    csv_end = time.time()

    print("\nReading radar pulses... done")
    print(f"Time taken to read CSV       : {csv_end - csv_start:.6f} seconds")
    print(f"Total Radar Readings Read    : {len(radar_df)}\n")

    #  COMPUTATION 
    
    compute_start = time.time()

    for i in range(0, len(radar_df), WINDOW_SIZE):
        process_window(radar_df.iloc[i:i + WINDOW_SIZE])

    compute_end = time.time()

    # OUTPUT (SHIP FORMAT)
  

    print(f"Detected Ships: {len(emitters)}")
    print("-" * 50)

    for idx, e in enumerate(emitters.values(), start=1):
        pw_median = np.median(e["pulse_widths"])
        print(
            f"ship{idx} : Pulses={e['pulse_count']}, "
            f"DOA = [MIN-{e['min_doa']:.2f},  MAX-{e['max_doa']:.2f}], "
            f"Freq[ MIN- {e['min_freq']:.2f}, MAX-  {e['max_freq']:.2f}], "
            f"PW={pw_median:.2f}"
        )

    print("-" * 50)
    print(f"Total Computation Time       : {compute_end - compute_start:.6f} seconds")
