import csv
import numpy as np
from sklearn.cluster import DBSCAN

# Read pulses from CSV
pulses = []
with open("radar_pulses.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        pulses.append([
            float(row["pulse_width"]),
            float(row["doa"]),
            float(row["pri"]),
            float(row["frequency"])
        ])

pulses = np.array(pulses)

# Dynamic tolerances: use standard deviation of each feature
std_dev = np.std(pulses, axis=0)
# Scale factor can be adjusted (0.5 works well for grouping similar ships)
tolerances = 0.5 * std_dev

print("Dynamic Tolerances:")
print(f"PW_TOL = {tolerances[0]:.2f}, DOA_TOL = {tolerances[1]:.2f}, "
      f"PRI_TOL = {tolerances[2]:.2f}, FREQ_TOL = {tolerances[3]:.2f}\n")

# DBSCAN clustering
# eps: maximum distance between points to be considered in same cluster
# Use mean of tolerances as eps (can be tuned if needed)
eps = np.mean(tolerances)
clustering = DBSCAN(eps=eps, min_samples=1).fit(pulses)

labels = clustering.labels_
num_ships = len(set(labels))

# Count pulses per detected ship
ships_detected = []
for ship_id in set(labels):
    count = np.sum(labels == ship_id)
    ships_detected.append((ship_id, count))

# Print output
print("Detected Ships:")
for ship_id, count in ships_detected:
    print(f"Ship {ship_id + 1}: Pulses = {count}")

print("\nTotal Emitting Ships:", num_ships)
