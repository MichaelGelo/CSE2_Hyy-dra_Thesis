import time
import csv
import threading
import subprocess
import statistics
import sys
from jtop import jtop

# ==========================================
# CONFIGURATION
# ==========================================
EXECUTABLE = "./finalcuda"  # Your compiled C++ program
POWER_LOG_FILE = "proof_power.csv"      # Evidence 1: The Power Data
OUTPUT_LOG_FILE = "proof_output.txt"    # Evidence 2: The Terminal Output

# Global control flags
keep_logging = True
power_readings = []
timestamps = []

def monitor_power():
    """
    Background thread that reads the INA3221 sensor 
    while the C++ code runs.
    """
    global keep_logging
    
    # We use jtop context manager to access sensors safely
    with jtop() as jetson:
        if not jetson.ok():
            print("[ERROR] JTOP is not running! Run with sudo.")
            return

        start_t = time.time()
        
        while keep_logging and jetson.ok():
            try:
                # Read Total System Power (ALL / VDD_IN)
                # jtop returns milliwatts, convert to Watts
                p_watts = jetson.stats['PowerTOT'] / 1000.0
                
                # Record timestamp relative to start
                current_t = time.time() - start_t
                
                power_readings.append(p_watts)
                timestamps.append(current_t)
            except Exception:
                pass # Ignore occasional sensor read errors
            
            # Sampling rate: 10 times per second (0.1s)
            # This is high resolution enough for the graph but low overhead
            time.sleep(0.1)

def run_benchmark():
    global keep_logging

    print(f"[{time.strftime('%H:%M:%S')}] Starting Benchmark Suite...")
    
    # 1. Start the Power Monitor in the background
    monitor_thread = threading.Thread(target=monitor_power)
    monitor_thread.start()
    
    # Allow sensor to stabilize for 1 second before launching workload
    time.sleep(1)
    
    # Clear "idle" readings so we only capture the run
    power_readings.clear()
    timestamps.clear()

    print(f"[{time.strftime('%H:%M:%S')}] Launching {EXECUTABLE}...")

    # 2. Run the C++ Program and Capture Output
    # We use Popen so we can stream the output to the screen AND a file simultaneously
    start_time = time.time()
    
    with open(OUTPUT_LOG_FILE, "w") as log_file:
        try:
            process = subprocess.Popen(
                [EXECUTABLE], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, # Merge errors into output
                text=True,
                bufsize=1
            )
            
            # Read output line by line as it happens
            for line in process.stdout:
                # Print to screen (for you to see)
                sys.stdout.write(line)
                # Write to proof file (for thesis evidence)
                log_file.write(line)
                
            process.wait() # Wait for C++ to finish
            
        except FileNotFoundError:
            print(f"[ERROR] Could not find {EXECUTABLE}. Did you compile it?")
            keep_logging = False
            return
        except Exception as e:
            print(f"[ERROR] Execution failed: {e}")
            keep_logging = False
            return

    end_time = time.time()
    total_duration = end_time - start_time
    
    # 3. Stop the Power Monitor
    keep_logging = False
    monitor_thread.join()
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Benchmark Complete.")

    # 4. Calculate and Save Results
    if len(power_readings) > 0:
        # Filter out any "zero" readings if sensor glitched
        valid_readings = [p for p in power_readings if p > 0]
        
        if not valid_readings:
            print("No valid power readings collected.")
            return

        avg_power = statistics.mean(valid_readings)
        peak_power = max(valid_readings)
        total_energy = avg_power * total_duration
        
        # Print Summary to Screen
        print("\n" + "="*50)
        print(f"       FINAL POWER REPORT (Time-Weighted)       ")
        print("="*50)
        print(f" Duration:            {total_duration:.4f} seconds")
        print(f" Average Active Power: {avg_power:.4f} Watts")
        print(f" Peak Power:           {peak_power:.4f} Watts")
        print(f" Total Energy:         {total_energy:.4f} Joules")
        print(f" Samples Collected:    {len(valid_readings)}")
        print("="*50)
        
        # Save Raw Data to CSV (This is your graph data)
        with open(POWER_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_Seconds", "Power_Watts"])
            for t, p in zip(timestamps, valid_readings):
                writer.writerow([f"{t:.2f}", f"{p:.2f}"])
                
        print(f"\n[EVIDENCE 1] Power log saved to: {POWER_LOG_FILE}")
        print(f"[EVIDENCE 2] Program output saved to: {OUTPUT_LOG_FILE}")
        
    else:
        print("[WARNING] No power readings were captured. Was the run too short?")

if __name__ == "__main__":
    run_benchmark()