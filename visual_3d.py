import csv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

num_ships = len(set(labels))
print("Detected Ships:", num_ships)

# --- 3D Visualization ---
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# X = Pulse Width, Y = DOA, Z = Frequency
scatter = ax.scatter(pulses[:,0], pulses[:,1], pulses[:,3], c=labels, cmap='rainbow', s=100, edgecolors='k')

ax.set_xlabel("Pulse Width (PW)")
ax.set_ylabel("DOA")
ax.set_zlabel("Frequency")
ax.set_title("Detected Ships Clustering (PW vs DOA vs Frequency)")
plt.colorbar(scatter, label="Ship Cluster ID")
plt.show()
