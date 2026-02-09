import socket
import struct
import time
import hyyrorunner  # Your separate module for Hyyro FPGA processing

SERVER_IP = '0.0.0.0'
PORT = 5000


def recv_all(sock, n):
    """Receive exactly n bytes from the socket."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def run_server():
    print(f"--- TCP Server listening on {SERVER_IP}:{PORT} ---")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((SERVER_IP, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    # --- Receive header ---
                    header = recv_all(conn, 8)
                    if not header:
                        print("Connection closed before header received")
                        continue

                    q_len, r_len = struct.unpack('ii', header)
                    print(f"[{addr}] Header received | Query length: {q_len}, Reference length: {r_len}")

                    # --- Receive query and reference ---
                    q_data = recv_all(conn, q_len)
                    if not q_data:
                        print(f"[{addr}] Connection closed before query received")
                        continue

                    r_data = recv_all(conn, r_len)
                    if not r_data:
                        print(f"[{addr}] Connection closed before reference received")
                        continue

                    # --- Call Hyyro module for FPGA processing ---
                    hw_start = time.time()
                    results = hyyrorunner.run_hyyro(q_data, r_data)
                    hw_exec_ms = (time.time() - hw_start) * 1000.0

                    # --- Format response ---
                    response = (f"Hardware Time: {results['hw_time_ms']} ms\n"
                                f"Final Score: {results['final_score']}\n"
                                f"Lowest Score: {results['lowest_score']}\n"
                                f"Indices: {results['indices']}\n")

                    conn.sendall(response.encode('ascii'))
                    print(f"[{addr}] Processed request | Score: {results['lowest_score']}")
                    print(f"Hardware Time: {results['hw_time_ms']} ms\n")

                except Exception as e:
                    print(f"[{addr}] Error: {e}")
                    try:
                        conn.sendall(f"Server Error: {str(e)}".encode('ascii'))
                    except:
                        pass  # ignore if sending fails


if __name__ == "__main__":
    run_server()


