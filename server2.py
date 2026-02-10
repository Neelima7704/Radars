# radar_server_tail.py
import socket
import pandas as pd
import pickle
import time
import os

HOST = "127.0.0.1"
PORT = 65432
WINDOW_SIZE = 10000
CSV_FILE = "DF5.csv"
POLL_INTERVAL = 0.2  # seconds

if __name__ == "__main__":
    last_sent_row = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = server.accept()
        with conn:
            print("Client connected:", addr)

            # Initial baseline
            df = pd.read_csv(CSV_FILE)
            last_sent_row = len(df)
            print(f"Initial rows detected: {last_sent_row}")

            while True:
                time.sleep(POLL_INTERVAL)

                # Re-read CSV to detect growth
                try:
                    df = pd.read_csv(CSV_FILE)
                except Exception as e:
                    print("CSV read error:", e)
                    continue

                current_rows = len(df)

                if current_rows <= last_sent_row:
                    continue  # no new data

                # Extract only NEW rows
                new_data = df.iloc[last_sent_row:current_rows]

                # Send in WINDOW_SIZE chunks
                for i in range(0, len(new_data), WINDOW_SIZE):
                    window_df = new_data.iloc[i:i + WINDOW_SIZE]

                    payload = pickle.dumps({
                        "csv_filename": CSV_FILE,
                        "data": window_df.to_dict("records")
                    })

                    conn.sendall(len(payload).to_bytes(8, "big") + payload)
                    print(f"Sent {len(window_df)} new pulses")

                last_sent_row = current_rows
