import csv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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

# Dynamic tolerances based on standard deviation
std_dev = np.std(pulses, axis=0)
tolerances = 0.5 * std_dev
eps = np.mean(tolerances)

# DBSCAN clustering
clustering = DBSCAN(eps=eps, min_samples=1).fit(pulses)
labels = clustering.labels_

# Print results
num_ships = len(set(labels))
print("Detected Ships:", num_ships)

# --- Visualization ---
plt.figure(figsize=(8,6))
scatter = plt.scatter(pulses[:,1], pulses[:,3], c=labels, cmap='rainbow', s=100, edgecolors='k')
plt.xlabel("DOA")
plt.ylabel("Frequency")
plt.title("Detected Ships Clustering (DOA vs Frequency)")
plt.grid(True)
plt.colorbar(scatter, label="Ship Cluster ID")
plt.show()
