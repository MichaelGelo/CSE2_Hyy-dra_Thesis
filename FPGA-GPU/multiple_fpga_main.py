import random
import time
from fpga_runner import run_fpga_test

ref_files = ["ref1_500k.fasta", "ref2_1M.fasta", "ref3_10M.fasta", "ref4_25M.fasta", "ref5_50M.fasta"]
que_files = ["que1_64.fasta", "que2_128.fasta", "que3_128.fasta", "que4_256.fasta", "que5_256.fasta"]

print("Starting continuous FPGA tests...")
print("Press Ctrl + C anytime to stop.\n")

try:
    while True:
        ref_choice = random.choice(ref_files)
        que_choice = random.choice(que_files)

        print("New random pair selected:")
        print(f"   Reference: {ref_choice}")
        print(f"   Query: {que_choice}")

        # --- Run FPGA test ---
        time_ms = run_fpga_test(ref_choice, que_choice)

        if time_ms is not None:
            print(f"Test complete — Execution time: {time_ms:.3f} ms")
        else:
            print("Test failed or no time returned.")

        print("-" * 50)
        # Optional: add a short delay between runs (to prevent overload)
        time.sleep(2)  # Adjust or remove if you want continuous fast testing

except KeyboardInterrupt:
    print("\nContinuous testing stopped by user.")