import time
import numpy as np
from pynq import Overlay, allocate

BITFILE_PATH = "/home/xilinx/jupyter_notebooks/updatedbit/DMA_wrappers.bit"

_ol = None
_hyyro = None
_dma0 = None
_dma1 = None


def init_overlay():
    global _ol, _hyyro, _dma0, _dma1

    print("[HYYRO] Loading overlay...")
    _ol = Overlay(BITFILE_PATH, download=False)
    _ol.download()
    print("[HYYRO] is_loaded:", _ol.is_loaded())
    print("[HYYRO] IPs:", list(_ol.ip_dict.keys()))
    _hyyro = _ol.hyyro_0
    _dma0 = _ol.axi_dma_0
    _dma1 = _ol.axi_dma_1
    print("[HYYRO] Overlay ready.\n")


def run_hyyro_buffers(query_buffer, ref_buffer):
    init_overlay()

    zero_idx_buffer = allocate(shape=(1024,), dtype=np.uint32)
    score_buffer = allocate(shape=(32,), dtype=np.uint32)

    try:
        _hyyro.write(0x10, int(len(query_buffer)))
        _hyyro.write(0x18, int(len(ref_buffer)))

        # arm output DMAs first
        _dma0.recvchannel.transfer(zero_idx_buffer)
        _dma1.recvchannel.transfer(score_buffer)

        hw_start = time.time()

        # send inputs
        _dma0.sendchannel.transfer(ref_buffer)
        _dma1.sendchannel.transfer(query_buffer)

        # start IP
        _hyyro.write(0x00, 0x01)

        # waits
        _dma0.sendchannel.wait()
        _dma1.sendchannel.wait()

        hw_exec_ms = (time.time() - hw_start) * 1000.0

        lowest_score = int(score_buffer[0])
        final_score = int(score_buffer[1])

        valid_indices = []
        for idx in zero_idx_buffer:
            v = int(idx)
            if v != 0 and v != 4294967295:
                valid_indices.append(v)

        return {
            "hw_time_ms": hw_exec_ms,
            "final_score": final_score,
            "lowest_score": lowest_score,
            "indices": valid_indices,
        }

    finally:
        zero_idx_buffer.close()
        score_buffer.close()