import random
from fpga_runner import run_fpga_test


ref_files = ["ref1_500k.fasta", "ref2_1M.fasta", "ref3_10M.fasta", "ref4_25M.fasta", "ref5_50M.fasta"]
que_files = ["que1_64.fasta", "que2_128.fasta", "que3_128.fasta", "que4_256.fasta", "que5_256.fasta"]


ref_choice = random.choice(ref_files)
que_choice = random.choice(que_files)

print(f"Selected random pair:")
print(f"   Reference: {ref_choice}")
print(f"   Query: {que_choice}\n")

# --- Run FPGA test ---
time_ms = run_fpga_test(ref_choice, que_choice)

if time_ms is not None:
    print(f"Test complete — execution time: {time_ms:.3f} ms")
else:
    print("Test failed or no time returned.")