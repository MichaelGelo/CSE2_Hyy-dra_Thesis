import time
import numpy as np
from pynq import Overlay, allocate
import subprocess
BITFILE_PATH = "/home/xilinx/jupyter_notebooks/updatedbit/DMA_wrappers.bit"
MAX_REF_LEN = 4 * 10**6 * 30

def free_mem():
    mem = subprocess.check_output("cat /proc/meminfo | grep 'MemFree'", shell=True).decode().strip()
    cma = subprocess.check_output("cat /proc/meminfo | grep 'CmaFree'", shell=True).decode().strip()
    
    print(f"    {mem}")
    print(f"    {cma}")
    
    # Parse and return MemFree value in kB
    mem_free_kb = int(mem.split()[1])
    return mem_free_kb

# --- CMA Memory Monitoring ---
def read_cma_bytes(debug=False):
    """Read total and free CMA memory from /proc/meminfo."""
    cma_total_bytes = None
    cma_free_bytes = None
    raw_total = None
    raw_free = None

    with open("/proc/meminfo", "r") as f:
        for line in f:
            s = line.lstrip()
            if s.startswith("CmaTotal:"):
                raw_total = line.rstrip("\n")
                cma_total_bytes = int(s.split()[1]) * 1024
            elif s.startswith("CmaFree:"):
                raw_free = line.rstrip("\n")
                cma_free_bytes = int(s.split()[1]) * 1024

    if cma_total_bytes is None or cma_free_bytes is None:
        raise RuntimeError("Could not read CmaTotal and CmaFree from /proc/meminfo")

    if debug:
        print("[CMA debug] raw lines:")
        print(" ", raw_total)
        print(" ", raw_free)

    return cma_total_bytes, cma_free_bytes


def get_cma_free_bytes(debug=False):
    """Get only free CMA memory in bytes."""
    _, free_b = read_cma_bytes(debug=debug)
    return free_b


def fmt_bytes_and_mb(x_bytes):
    """Format bytes as both bytes and MB."""
    return f"{x_bytes} bytes ({x_bytes / (1024 * 1024):.2f} MB)"


def run_hyyro(query_bytes, ref_bytes, cma_free_before):
    """Execute FPGA kernel with query and reference sequences.
    
    Monitors CMA memory allocation before and after execution.
    """
    
    # Load overlay
    ol = Overlay(BITFILE_PATH)
    hyyro_ip = ol.hyyro_0

    # DMA engines
    dma_ref = ol.axi_dma_0
    dma_query = ol.axi_dma_1
    dma_zero_idx = ol.axi_dma_0
    dma_score = ol.axi_dma_1
    print("[CMA AND MEM] [BEFORE ALLOCATION]")
    mem_free_before_kb = free_mem()
    # Allocate buffers
    ref_buffer = allocate(shape=(len(ref_bytes),), dtype=np.uint8)
    query_buffer = allocate(shape=(len(query_bytes),), dtype=np.uint8)
    zero_idx_buffer = allocate(shape=(1024,), dtype=np.uint32)
    score_buffer = allocate(shape=(32,), dtype=np.uint32)
    print("[CMA AND MEM] [AFTER ALLOCATION]")
    mem_free_after_kb = free_mem()
    
    # Calculate MemFree difference
    mem_diff_kb = mem_free_before_kb - mem_free_after_kb
    mem_diff_bytes = mem_diff_kb * 1024
    mem_diff_mb = mem_diff_bytes / (1024 * 1024)
    print(f"\n[MEMFREE DIFFERENCE] {mem_diff_kb} kB → {mem_diff_bytes} bytes → {mem_diff_mb:.2f} MB")
    ref_minus_mem = mem_diff_bytes - len(ref_bytes)
    print(f"[MEMFREE DIFFERENCE - REFERENCE LENGTH] = {ref_minus_mem} bytes\n")
    # Record CMA memory after allocation
    cma_free_after_alloc = get_cma_free_bytes(debug=False)
    cma_used_alloc = cma_free_before - cma_free_after_alloc

    print("[CMA ALLOCATION SUMMARY] [HYYRORUNNER]")
    print(f"  CMA free BEFORE: {fmt_bytes_and_mb(cma_free_before)}")
    print(f"  CMA free AFTER : {fmt_bytes_and_mb(cma_free_after_alloc)}")
    print(f"  CMA USED       : {fmt_bytes_and_mb(cma_used_alloc)}\n")

    # Copy query to buffer
    np.copyto(query_buffer, np.frombuffer(query_bytes, dtype=np.uint8))

    # Copy reference to buffer
    np.copyto(
        ref_buffer[:len(ref_bytes)],
        np.frombuffer(ref_bytes, dtype=np.uint8)
    )

    # Start FPGA timing
    hw_start = time.time()

    # Write lengths to IP
    hyyro_ip.write(0x10, len(query_bytes))
    hyyro_ip.write(0x18, len(ref_bytes))

    # Prepare DMA receive channels
    dma_zero_idx.recvchannel.transfer(zero_idx_buffer)
    dma_score.recvchannel.transfer(score_buffer)

    # Send buffers
    dma_ref.sendchannel.transfer(ref_buffer[:len(ref_bytes)])
    dma_query.sendchannel.transfer(query_buffer)

    # Start kernel
    hyyro_ip.write(0x00, 0x01)

    # Wait for completion
    dma_ref.sendchannel.wait()
    dma_query.sendchannel.wait()

    # Stop kernel
    hyyro_ip.write(0x00, 0x00)

    # Compute hardware execution time
    hw_exec_ms = (time.time() - hw_start) * 1000.0

    # Extract results
    final_score = int(score_buffer[1])
    lowest_score = int(score_buffer[0])

    valid_indices = [
        int(idx)
        for idx in zero_idx_buffer
        if idx not in (0, 4294967295)
    ]

    # Free buffers
    ref_buffer.close()
    query_buffer.close()
    zero_idx_buffer.close()
    score_buffer.close()

    del ref_buffer
    del query_buffer
    del zero_idx_buffer
    del score_buffer
    print("[CMA AND MEM] [AFTER DEALLOCATION]")
    free_mem()

    print(f"FPGA TIME: {hw_exec_ms:.2f} ms")
    print(f"FINAL SCORE: {final_score}")
    print(f"LOWEST SCORE: {lowest_score}")
    print(f"VALID INDICES: {valid_indices[:5]}")
    print()

    return {
        "hw_time_ms": hw_exec_ms,
        "final_score": final_score,
        "lowest_score": lowest_score,
        "indices": valid_indices,
        "cma_alloc_bytes": cma_used_alloc,
        "cma_free_before": cma_free_before
    }