from pynq import Overlay
import time

# CONFIRM THIS PATH IS CORRECT
OVERLAY_PATH = "/home/xilinx/jupyter_notebooks/updatedbit/DMA_wrapper.bit"

def force_reset(name, dma):
    print(f"--- RESETTING {name} ---")
    try:
        # 1. Stop (Write 0 to Control)
        dma.sendchannel._mmio.write(0x00, 0)
        dma.recvchannel._mmio.write(0x30, 0)
        # 2. Reset (Write 1 to Reset Bit)
        dma.sendchannel._mmio.write(0x00, 4)
        dma.recvchannel._mmio.write(0x30, 4)
        time.sleep(1)
        # 3. Check Status (Bit 0 of Status Reg should be 1 = Halted)
        send_status = dma.sendchannel._mmio.read(0x04)
        recv_status = dma.recvchannel._mmio.read(0x34)

        if (send_status & 1) and (recv_status & 1):
            print(f"✅ {name} is FIXED and HALTED.")
        else:
            print(f"❌ {name} is STILL STUCK. (Send: {hex(send_status)}, Recv: {hex(recv_status)})")
    except Exception as e:
        print(f"Could not reset {name}: {e}")

ol = Overlay(OVERLAY_PATH)
if 'axi_dma_0' in ol.ip_dict: force_reset('axi_dma_0', ol.axi_dma_0)
if 'axi_dma_1' in ol.ip_dict: force_reset('axi_dma_1', ol.axi_dma_1)