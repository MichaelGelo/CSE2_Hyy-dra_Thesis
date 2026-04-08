import socket
import struct
import subprocess
import numpy as np
from pynq import allocate
import hyyrorunner
import time


print("[SERVER] Initializing FPGA overlay...")
hyyrorunner.init_overlay()
print("[SERVER] FPGA ready.\n")

SERVER_IP = "0.0.0.0"
PORT = 5000

MAX_QUERY_SIZE = 1024
MAX_REFERENCE_SIZE = 110 * 1024 * 1024


def free_mem_kb():
    memfree_kb = int(subprocess.check_output(
        "awk '/MemFree:/ {print $2}' /proc/meminfo", shell=True
    ).decode().strip())

    cmafree_kb = int(subprocess.check_output(
        "awk '/CmaFree:/ {print $2}' /proc/meminfo", shell=True
    ).decode().strip())

    print(f"    MemFree: {memfree_kb} kB")
    print(f"    CmaFree: {cmafree_kb} kB")

    return memfree_kb, cmafree_kb


def recv_exact_into(sock, size, view):
    recvd = 0
    chunk_size = 256 * 1024
    while recvd < size:
        to_recv = min(chunk_size, size - recvd)
        n = sock.recv_into(view[recvd:recvd + to_recv], to_recv)
        if n == 0:
            return False
        recvd += n
    return True


def run_server():
    print(f"--- Optimized FPGA Server on {SERVER_IP}:{PORT} ---")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024 * 1024)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024 * 1024)

        s.bind((SERVER_IP, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024 * 1024)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024 * 1024)

            with conn:
                query_cma = None
                ref_cma = None

                try:
                    print(f"[{addr}] Connected")

                    header = conn.recv(8)
                    if not header:
                        continue

                    q_len, r_len = struct.unpack("ii", header)

                    print(f"[RECEIVE] Query: {q_len} bytes, Reference: {r_len / (1024 * 1024):.2f} MB")

                    if q_len > MAX_QUERY_SIZE or r_len > MAX_REFERENCE_SIZE:
                        raise ValueError("Incoming data exceeds allowed size")

                    print("[FPGA] Allocating CMA buffers...")

                    print("[MEM] Before allocate:")
                    mem_before_kb, cma_before_kb = free_mem_kb()

                    query_cma = allocate(shape=(q_len,), dtype=np.uint8)
                    ref_cma = allocate(shape=(r_len,), dtype=np.uint8)

                    print("[MEM] After allocate:")
                    mem_after_kb, cma_after_kb = free_mem_kb()

                    mem_used_kb = mem_before_kb - mem_after_kb
                    cma_used_kb = cma_before_kb - cma_after_kb

                    print(f"[MEM] Used MemFree: {mem_used_kb} kB ({mem_used_kb / 1024.0:.2f} MB)")
                    print(f"[MEM] Used CmaFree: {cma_used_kb} kB ({cma_used_kb / 1024.0:.2f} MB)\n")

                    recv_start = time.time()

                    if not recv_exact_into(conn, q_len, memoryview(query_cma)):
                        raise ConnectionError("Query receive failed")

                    if not recv_exact_into(conn, r_len, memoryview(ref_cma)):
                        raise ConnectionError("Reference receive failed")

                    recv_time = time.time() - recv_start
                    total_data = q_len + r_len
                    recv_mbps = (total_data * 8.0) / (recv_time * 1e6)

                    print(f"[RECEIVE] {q_len} + {r_len} = {total_data / (1024 * 1024):.2f} MB total")
                    print(f"[RECEIVE] Time: {recv_time:.4f}s -> {recv_mbps:.2f} Mbps\n")

                    hw_start_total = time.time()
                    results = hyyrorunner.run_hyyro_buffers(query_cma, ref_cma)
                    hw_total_time_sec = time.time() - hw_start_total

                    response = (
                        f"Hardware Time: {results['hw_time_ms']} ms\n"
                        f"Final Score: {results['final_score']}\n"
                        f"Lowest Score: {results['lowest_score']}\n"
                        f"Indices: {results['indices']}\n"
                    )

                    print(f"[{addr}] Hardware exec time: {results['hw_time_ms']:.2f} ms")
                    print(f"[{addr}] Lowest score     : {results['lowest_score']}")
                    print(f"[{addr}] Total time       : {hw_total_time_sec * 1000:.2f} ms\n")

                    conn.sendall(response.encode("ascii"))
                    print("=" * 57)

                except Exception as e:
                    print(f"[{addr}] Error: {e}")
                    try:
                        conn.sendall(f"Server Error: {str(e)}".encode("ascii"))
                    except Exception:
                        pass

                finally:
                    if ref_cma is not None:
                        try:
                            ref_cma.close()
                        except Exception:
                            pass
                        ref_cma = None

                    if query_cma is not None:
                        try:
                            query_cma.close()
                        except Exception:
                            pass
                        query_cma = None


if __name__ == "__main__":
    run_server()