import csv
import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

pulses = []

csv_start_time = time.time()

with open("radar_pulses.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        pulses.append([
            float(row["pulse_width"]),   # PW
            float(row["doa"]),           # DOA
            float(row["pri"]),           # PRI
            float(row["frequency"])      # Frequency
        ])

csv_end_time = time.time()
csv_read_time = csv_end_time - csv_start_time

pulses = np.array(pulses)

print("\nReading radar pulses... done")
print(f"Time taken to read CSV       : {csv_read_time:.6f} seconds")
print(f"Total Radar Readings Read    : {pulses.shape[0]}")


# 2. COMPUTATION


compute_start_time = time.time()

# Dynamic tolerances 
PW_TOL   = np.std(pulses[:, 0])
DOA_TOL  = np.std(pulses[:, 1])
PRI_TOL  = np.std(pulses[:, 2])
FREQ_TOL = np.std(pulses[:, 3])

# Feature scaling 
scaler = StandardScaler()
pulses_scaled = scaler.fit_transform(pulses)

# Dynamic EPS calculation 
std_dev_scaled = np.std(pulses_scaled, axis=0)
eps = np.mean(std_dev_scaled) * 0.6

# DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=3)
labels = dbscan.fit_predict(pulses_scaled)

# Extract valid clusters 
valid_labels = set(labels)
valid_labels.discard(-1)

#Aggregate ship parameters
ship_outputs = {}
ship_index = 1

for label in sorted(valid_labels):
    ship_pulses = pulses[labels == label]

    min_doa  = np.min(ship_pulses[:, 1])
    max_doa  = np.max(ship_pulses[:, 1])
    min_freq = np.min(ship_pulses[:, 3])
    max_freq = np.max(ship_pulses[:, 3])
    pw       = np.median(ship_pulses[:, 0])

    ship_outputs[f"ship{ship_index}"] = (
        min_doa, max_doa, min_freq, max_freq, pw, len(ship_pulses)
    )

    ship_index += 1

compute_end_time = time.time()
compute_time = compute_end_time - compute_start_time


# 3. OUTPUT


print("\nDetected Ships:", len(ship_outputs))
print("-" * 50)

for ship, values in ship_outputs.items():
    min_doa, max_doa, min_freq, max_freq, pw, count = values
    print(
        f"{ship} : Pulses={count}, "
        f"DOA[{min_doa:.2f}, {max_doa:.2f}], "
        f"Freq[{min_freq:.2f}, {max_freq:.2f}], "
        f"PW={pw:.2f}"
    )
    
print("-" * 50)


print(f"Total Computation Time       : {compute_time:.6f} seconds")

