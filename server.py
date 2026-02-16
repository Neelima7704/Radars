# radar_server.py
import socket
import pandas as pd
import pickle
import time
import os

HOST = "127.0.0.1"
PORT = 65432
WINDOW_SIZE = 10000
DATA_DIRECTORY = "DATA"   # <<< PUT YOUR FOLDER NAME HERE


def stream_directory(conn):
    t_index = 0

    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIRECTORY, filename)
            print(f"Reading file: {filename}")

            df = pd.read_csv(file_path)

            for i in range(0, len(df), WINDOW_SIZE):
                window_df = df.iloc[i:i + WINDOW_SIZE]
                payload = pickle.dumps(window_df.to_dict("records"))

                conn.sendall(len(payload).to_bytes(8, "big") + payload)
                t_index += 1
                print(f"t{t_index} → sent {len(window_df)} pulses")

                time.sleep(0.005)


if __name__ == "__main__":

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = server.accept()

        with conn:
            print("Client connected:", addr)

            # FIRST PASS
            print("---- First Directory Read ----")
            stream_directory(conn)

            # SECOND PASS (Re-read)
            print("---- Second Directory Read ----")
            stream_directory(conn)

            print("All data sent. Closing connection.")
