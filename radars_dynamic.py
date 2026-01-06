import csv
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# 1. Read radar pulses from CSV

pulses = []

with open("radar_pulses.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        pulses.append([
            float(row["pulse_width"]),   # PW
            float(row["doa"]),           # DOA
            float(row["pri"]),           # PRI
            float(row["frequency"])      # Frequency
        ])

pulses = np.array(pulses)

# 2. Calculate dynamic tolerances (RAW DATA)

PW_TOL   = np.std(pulses[:, 0])
DOA_TOL  = np.std(pulses[:, 1])
PRI_TOL  = np.std(pulses[:, 2])
FREQ_TOL = np.std(pulses[:, 3])

print(f"\nPW_TOL = {PW_TOL:.2f}, "
      f"DOA_TOL = {DOA_TOL:.2f}, "
      f"PRI_TOL = {PRI_TOL:.2f}, "
      f"FREQ_TOL = {FREQ_TOL:.2f}")


# 3. Scale features (important for DBSCAN)

scaler = StandardScaler()
pulses_scaled = scaler.fit_transform(pulses)


# 4. Dynamic eps calculation

std_dev_scaled = np.std(pulses_scaled, axis=0)
eps = np.mean(std_dev_scaled) * 0.6   # tunable factor


# 5. DBSCAN clustering

dbscan = DBSCAN(
    eps=eps,
    min_samples=3
)

labels = dbscan.fit_predict(pulses_scaled)


# 6. Extract valid ship clusters

valid_labels = set(labels)
valid_labels.discard(-1)


# 7. Aggregate ship parameters

ship_outputs = {}
ship_index = 1

print("\nDetected Ships:")

for label in valid_labels:
    ship_pulses = pulses[labels == label]
    pulse_count = len(ship_pulses)

    print(f"Ship {ship_index}: Pulses = {pulse_count}")

    min_doa  = np.min(ship_pulses[:, 1])
    max_doa  = np.max(ship_pulses[:, 1])
    min_freq = np.min(ship_pulses[:, 3])
    max_freq = np.max(ship_pulses[:, 3])
    pw       = np.median(ship_pulses[:, 0])

    ship_outputs[f"ship{ship_index}"] = (
        min_doa, max_doa, min_freq, max_freq, pw
    )

    ship_index += 1


# 8. Detailed ship parameters output

print("\n" + "-" * 60)

for ship, values in ship_outputs.items():
    min_doa, max_doa, min_freq, max_freq, pw = values

    print(
        f"{ship} : ["
        f"min_doa : {min_doa:.2f}, "
        f"max_doa : {max_doa:.2f}, "
        f"min_freq : {min_freq:.2f}, "
        f"max_freq : {max_freq:.2f}, "
        f"pw : {pw:.2f}"
        f"]"
    )
