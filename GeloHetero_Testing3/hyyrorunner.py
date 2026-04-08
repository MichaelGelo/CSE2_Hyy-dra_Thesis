import time
from pynq import Overlay, allocate
import numpy as np




def run_hyyro_buffers(query_buffer, ref_buffer):
    
    BITFILE_PATH = "/home/xilinx/jupyter_notebooks/updatedbit/DMA_wrappers.bit"

    print("[HYYRO] Loading overlay...")
    ol = Overlay(BITFILE_PATH)
    hyyro_ip = ol.hyyro_0

    dma_ref = ol.axi_dma_0
    dma_query = ol.axi_dma_1
    dma_zero_idx = ol.axi_dma_0
    dma_score = ol.axi_dma_1
    print("[HYYRO] Overlay ready.\n")

    zero_idx_buffer = allocate(shape=(1024,), dtype=np.uint32)
    score_buffer = allocate(shape=(32,), dtype=np.uint32)

    hyyro_ip.write(0x10, len(query_buffer))
    hyyro_ip.write(0x18, len(ref_buffer))

    dma_zero_idx.recvchannel.transfer(zero_idx_buffer)
    dma_score.recvchannel.transfer(score_buffer)

    hw_start = time.time()

    dma_ref.sendchannel.transfer(ref_buffer)
    dma_query.sendchannel.transfer(query_buffer)

    hyyro_ip.write(0x00, 0x01)

    dma_ref.sendchannel.wait()
    dma_query.sendchannel.wait()

    hyyro_ip.write(0x00, 0x00)

    hw_exec_ms = (time.time() - hw_start) * 1000.0


    final_score = int(score_buffer[1])
    lowest_score = int(score_buffer[0])

    valid_indices = [
        int(idx)
        for idx in zero_idx_buffer
        if idx not in (0, 4294967295)
    ]

    zero_idx_buffer.close()
    score_buffer.close()

    return {
        "hw_time_ms": hw_exec_ms,
        "final_score": final_score,
        "lowest_score": lowest_score,
        "indices": valid_indices,
    }