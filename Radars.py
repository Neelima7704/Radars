<pre>
import csv

# REQUIRED TOLERANCES
PW_TOL = 0.5
DOA_TOL = 2.5
PRI_TOL = 1.0
FREQ_TOL = 6.0

ships_detected = []

def is_same_ship(ship, pulse):
    pw, doa, pri, freq, count = ship
    pw2, doa2, pri2, freq2 = pulse

    return (
        abs(pw - pw2) <= PW_TOL and
        abs(doa - doa2) <= DOA_TOL and
        abs(pri - pri2) <= PRI_TOL and
        abs(freq - freq2) <= FREQ_TOL
    )

with open("radar_pulses.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        pulse = (
            float(row["pulse_width"]),
            float(row["doa"]),
            float(row["pri"]),
            float(row["frequency"])
        )

        matched = False
        for ship in ships_detected:
            if is_same_ship(ship, pulse):
                ship[4] += 1
                matched = True
                break

        if not matched:
            ships_detected.append([
                pulse[0], pulse[1], pulse[2], pulse[3], 1
            ])

# PRINT OUTPUT
print("Detected Ships:")
for i, ship in enumerate(ships_detected, 1):
    print(f"Ship {i}: Pulses = {ship[4]}")

print("\nTotal Emitting Ships:", len(ships_detected))



</pre>
