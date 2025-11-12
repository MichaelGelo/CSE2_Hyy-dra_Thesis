import random
import os
import time
import ctypes
import threading
from ctypes import c_char_p, c_int, byref

# --- EXTERNAL DEPENDENCIES ---
from fpga_runner import run_fpga_test

# --- CUDA LIBRARY SETUP (FIXED FOR UBUNTU) ---
LIBRARY_NAME = 'leven_wrapper.so'  # Use .so for Linux shared libraries
try:
    leven_lib = ctypes.CDLL(os.path.abspath(LIBRARY_NAME))
    leven_lib.align_sequences_gpu.argtypes = [
        c_char_p, c_int, c_char_p, c_int,
        ctypes.POINTER(c_int), ctypes.POINTER(c_int), ctypes.POINTER(c_int)
    ]
    leven_lib.align_sequences_gpu.restype = c_int
except OSError:
    print(f"Error: Could not load CUDA library '{LIBRARY_NAME}'. Ensure it is compiled.")
    exit()
# ---------------------------------------------


# --- Global Results Dictionary for Thread Communication ---
RESULTS = {
    'gpu_time': None,
    'fpga_time': None,
    'gpu_score': None,
    'gpu_index': None,
    'gpu_last': None,
    'fpga_score': None,
}
# -----------------------------------------------------------


def parse_fasta(filepath):
    """Reads a FASTA file and returns sequences as a dictionary."""
    sequences = {}
    current_header = None
    current_seq = []

    if not os.path.exists(filepath):
        print(f"Error: FASTA file not found at {filepath}")
        return {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            elif current_header:
                current_seq.append(line.upper().replace(' ', ''))

        if current_header and current_seq:
            sequences[current_header] = "".join(current_seq)

    return sequences


def save_fasta_part(filename, header, sequence):
    """Saves a sequence part to a new FASTA file."""
    with open(filename, 'w') as f:
        f.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i + 80] + '\n')


# --- THREAD WORKER 1: GPU ALIGNMENT ---
def gpu_worker(q_bytes, q_len, r1_bytes, r1_len):
    """Worker function for running the GPU alignment."""
    print("--- GPU THREAD: Started Alignment (Part 1) ---")

    lowest_score = c_int(0)
    lowest_index = c_int(-1)
    last_score = c_int(0)

    start_gpu = time.perf_counter()
    status = leven_lib.align_sequences_gpu(
        q_bytes, q_len, r1_bytes, r1_len,
        byref(lowest_score), byref(lowest_index), byref(last_score)
    )
    gpu_time = (time.perf_counter() - start_gpu) * 1000  # Convert to ms

    if status == 0:
        RESULTS['gpu_time'] = gpu_time
        RESULTS['gpu_score'] = lowest_score.value
        RESULTS['gpu_index'] = lowest_index.value
        RESULTS['gpu_last'] = last_score.value
        print(f"--- GPU THREAD: Finished in {gpu_time:.3f} ms ---")
    else:
        print(f"--- GPU THREAD: FAILED with Status Code: {status} ---")


# --- THREAD WORKER 2: FPGA TEST ---
def fpga_worker(fpga_temp_file, que_choice_name):
    """Worker function for running the FPGA test."""
    print("--- FPGA THREAD: Started Test (Part 2) ---")
    print(f"Attempting to run FPGA with files:")
    print(f"  Ref: {fpga_temp_file}")
    print(f"  Que: {que_choice_name}")

    fpga_time = run_fpga_test(fpga_temp_file, que_choice_name)

    if fpga_time is not None:
        RESULTS['fpga_time'] = fpga_time
        print(f"--- FPGA THREAD: Finished in {fpga_time:.3f} ms ---")
    else:
        print("--- FPGA THREAD: Test failed or no time returned. ---")


def run_hybrid_split_test():
    """Splits reference and runs GPU and FPGA concurrently."""
    global RESULTS

    ref_files = ["ref1_500k.fasta", "ref2_1M.fasta", "ref3_10M.fasta", "ref4_25M.fasta", "ref5_50M.fasta"]
    que_files = ["que1_64.fasta", "que2_128.fasta", "que3_128.fasta", "que4_256.fasta", "que5_256.fasta"]
    FPGA_TEMP_FILE = "Ref_Part2_FPGA.fasta"

    ref_choice = random.choice(ref_files)
    que_choice_name = random.choice(que_files)

    print(f"--- Selected Pair ---")
    print(f" Reference: {ref_choice}")
    print(f" Query: {que_choice_name}")
    print("---------------------\n")

    queries = parse_fasta(que_choice_name)
    references = parse_fasta(ref_choice)
    if not queries or not references:
        print("Error: Could not load data files.")
        return

    q_id, QUERY_SEQ = next(iter(queries.items()))
    q_bytes = QUERY_SEQ.encode('ascii')
    q_len = len(q_bytes)

    ref_id, REF_SEQ = next(iter(references.items()))
    r_len = len(REF_SEQ)

    split_point = r_len // 2
    REF_PART1 = REF_SEQ[:split_point]
    REF_PART2 = REF_SEQ[split_point:]

    r1_len = len(REF_PART1)
    r2_len = len(REF_PART2)

    r1_bytes = REF_PART1.encode('ascii')

    print(f"Total Reference Length: {r_len:,} bases")
    print(f"-> GPU Reference (Part 1) Length: {r1_len:,} bases")
    print(f"-> FPGA Reference (Part 2) Length: {r2_len:,} bases")

    save_fasta_part(FPGA_TEMP_FILE, f"{ref_id}_PART2", REF_PART2)

    print("\n==================================")
    print("STARTING CONCURRENT EXECUTION")
    print("==================================")

    start_concurrent = time.perf_counter()

    gpu_thread = threading.Thread(target=gpu_worker, args=(q_bytes, q_len, r1_bytes, r1_len))
    fpga_thread = threading.Thread(target=fpga_worker, args=(FPGA_TEMP_FILE, que_choice_name))

    gpu_thread.start()
    fpga_thread.start()

    gpu_thread.join()
    fpga_thread.join()

    end_concurrent = time.perf_counter()
    total_concurrent_time = (end_concurrent - start_concurrent) * 1000

    print("\n==================================")
    print("CONCURRENT RESULTS SUMMARY")
    print(f"Total Reference: {ref_choice} ({r_len:,} bases)")
    print("----------------------------------")

    gpu_time = RESULTS.get('gpu_time')
    fpga_time = RESULTS.get('fpga_time')

    if gpu_time is not None:
        print(f"GPU (Part 1) Time: {gpu_time:.3f} ms")
        print(f"GPU Result: Score={RESULTS.get('gpu_score')}, Index={RESULTS.get('gpu_index')}, Last={RESULTS.get('gpu_last')}")
    if fpga_time is not None:
        print(f"FPGA (Part 2) Time: {fpga_time:.3f} ms")

    if gpu_time is not None and fpga_time is not None:
        bottleneck_time = max(gpu_time, fpga_time)
        print(f"\nTime Bottleneck: {bottleneck_time:.3f} ms")
        print(f"Total Elapsed Concurrent Time: {total_concurrent_time:.3f} ms")
        print("(Total Elapsed Time should be close to the Bottleneck Time)")

    print("==================================")

    os.remove(FPGA_TEMP_FILE)
    print(f"Cleaned up temporary file: {FPGA_TEMP_FILE}")


if __name__ == '__main__':
    run_hybrid_split_test()
