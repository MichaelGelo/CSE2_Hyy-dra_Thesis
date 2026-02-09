import time
import numpy as np
from pynq import Overlay, allocate

BITFILE_PATH = "/home/xilinx/jupyter_notebooks/updatedbit/DMA_wrappers.bit"
MAX_REF_LEN = 4 * 10**6 * 30

def run_hyyro(query_bytes, ref_bytes):
    ol = Overlay(BITFILE_PATH)
    hyyro_ip = ol.hyyro_0
    dma_ref = ol.axi_dma_0
    dma_query = ol.axi_dma_1
    dma_zero_idx = ol.axi_dma_0
    dma_score = ol.axi_dma_1

    # Allocate buffers
    ref_buffer = allocate(shape=(MAX_REF_LEN,), dtype=np.uint8)
    query_buffer = allocate(shape=(len(query_bytes),), dtype=np.uint8)
    zero_idx_buffer = allocate((1024,), dtype=np.uint32)
    score_buffer = allocate((32,), dtype=np.uint32)

    # Copy data
    np.copyto(query_buffer, np.frombuffer(query_bytes, dtype=np.uint8))
    np.copyto(ref_buffer[:len(ref_bytes)], np.frombuffer(ref_bytes, dtype=np.uint8))

    # Start hardware
    hw_start = time.time()
    hyyro_ip.write(0x10, len(query_bytes))
    hyyro_ip.write(0x18, len(ref_bytes))

    dma_zero_idx.recvchannel.transfer(zero_idx_buffer)
    dma_score.recvchannel.transfer(score_buffer)
    dma_ref.sendchannel.transfer(ref_buffer[:len(ref_bytes)])
    dma_query.sendchannel.transfer(query_buffer)
    hyyro_ip.write(0x00, 0x01)
    dma_ref.sendchannel.wait()
    dma_query.sendchannel.wait()
    hyyro_ip.write(0x00, 0x00)

    hw_exec_ms = (time.time() - hw_start) * 1000.0
    final_score = int(score_buffer[1])
    lowest_score = int(score_buffer[0])
    valid_indices = [int(idx) for idx in zero_idx_buffer if idx not in (0, 4294967295)]
    
    ref_buffer.close()
    query_buffer.close()
    zero_idx_buffer.close()
    score_buffer.close()
    
    print(f"FPGA TIME: {hw_exec_ms:.2f} ms\n")

    return {
        "hw_time_ms": hw_exec_ms,
        "final_score": final_score,
        "lowest_score": lowest_score,
        "indices": valid_indices
    }
