# server_udp_fast.py
import socket
import pandas as pd
import pickle
import math

HOST = "127.0.0.1"
PORT = 65432
WINDOW_SIZE = 2000
CHUNK_SIZE = 60000

if __name__ == "__main__":

    df = pd.read_csv("DF5.csv")

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((HOST, PORT))

    print(f"Server listening on {HOST}:{PORT}")
    msg, addr = server.recvfrom(1024)
    print("Client connected:", addr)

    for t_index, i in enumerate(range(0, len(df), WINDOW_SIZE), start=1):

        window_df = df.iloc[i:i + WINDOW_SIZE]
        payload = pickle.dumps(window_df.to_dict("records"))

        total_chunks = math.ceil(len(payload) / CHUNK_SIZE)

        for chunk_id in range(total_chunks):
            start = chunk_id * CHUNK_SIZE
            chunk = payload[start:start + CHUNK_SIZE]

            header = f"{chunk_id}/{total_chunks}|".encode()
            server.sendto(header + chunk, addr)

        print(f"t{t_index} → sent {len(window_df)} pulses")

    server.sendto(b"END", addr)
    print("All data sent.")
