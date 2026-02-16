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
EXECUTABLE = "./finalcuda"
# Make sure this path exists!
POWER_LOG_FILE = "/home/dlsu-cse/power_logs/logs/test.csv" 
OUTPUT_LOG_FILE = "proof_output.txt"

# Global control flags
keep_logging = True
power_readings = []
timestamps = []

def monitor_power():
    global keep_logging
    
    print("[Power Monitor] Connecting to sensors...")
    with jtop() as jetson:
        if not jetson.ok():
            print("[ERROR] JTOP failed to start! (Did you run with sudo?)")
            return

        # --- AUTO-DETECT POWER KEY ---
        possible_keys = [
            'Power TOT', 'PowerTOT', 'ALL', 
            'Power 1-00081', 'POM_5V_IN', 'VDD_IN'
        ]
        
        power_key = None
        stats = jetson.stats
        for key in possible_keys:
            if key in stats:
                power_key = key
                break
        
        if power_key is None:
            print(f"[CRITICAL ERROR] Could not find a valid power rail!")
            return

        print(f"[Power Monitor] Success! Logging rail: '{power_key}'")

        start_t = time.time()
        
        while keep_logging and jetson.ok():
            try:
                val = jetson.stats[power_key]
                
                if isinstance(val, dict):
                    val = val.get('cur', val.get('avg', 0))

                if isinstance(val, str):
                    try:
                        val = float(val.replace('mW', '').strip())
                    except:
                        val = 0

                if isinstance(val, (int, float)):
                    if val > 100: 
                        p_watts = val / 1000.0
                    else:
                        p_watts = val
                    
                    power_readings.append(p_watts)
                    timestamps.append(time.time() - start_t)
            except Exception as e:
                pass 
            
            time.sleep(0.1)

def run_benchmark():
    global keep_logging

    print(f"[{time.strftime('%H:%M:%S')}] Starting Benchmark Suite...")
    
    monitor_thread = threading.Thread(target=monitor_power)
    monitor_thread.start()
    time.sleep(1.5) 
    
    power_readings.clear()
    timestamps.clear()

    print(f"[{time.strftime('%H:%M:%S')}] Launching {EXECUTABLE}...")

    start_time = time.time()
    with open(OUTPUT_LOG_FILE, "w") as log_file:
        try:
            process = subprocess.Popen(
                [EXECUTABLE], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            process.wait()
        except Exception as e:
            print(f"[ERROR] Execution failed: {e}")
            keep_logging = False
            return

    end_time = time.time()
    total_duration = end_time - start_time
    
    keep_logging = False
    monitor_thread.join()
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Benchmark Complete.")

    if len(power_readings) > 0:
        valid_readings = [p for p in power_readings if p > 0]
        if not valid_readings:
            print("[WARNING] Readings collected but all were 0W.")
            return

        avg_power = statistics.mean(valid_readings)
        peak_power = max(valid_readings)
        total_energy = avg_power * total_duration
        
        print("\n" + "="*50)
        print(f"       FINAL RESULTS (Thesis Evidence)       ")
        print("="*50)
        print(f" Duration:            {total_duration:.4f} s")
        print(f" Average Total Power: {avg_power:.4f} W")
        print(f" Peak Power:          {peak_power:.4f} W")
        print(f" Total Energy:        {total_energy:.4f} J")
        print("="*50)
        
        # =========================================================
        # MODIFIED CSV WRITER: SAVES RAW DATA AND SUMMARY STATS
        # =========================================================
        with open(POWER_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 1. Write the Headers
            writer.writerow(["Time_s", "Power_W"])
            
            # 2. Write the Raw Logs
            for t, p in zip(timestamps, valid_readings):
                writer.writerow([f"{t:.2f}", f"{p:.2f}"])
            
            # 3. Write an empty row for separation
            writer.writerow([])
            writer.writerow([])
            
            # 4. Write the Summary Statistics Table
            writer.writerow(["---", "SUMMARY STATISTICS", "---"])
            writer.writerow(["Metric", "Value", "Unit"])
            writer.writerow(["Duration", f"{total_duration:.4f}", "s"])
            writer.writerow(["Average Power", f"{avg_power:.4f}", "W"])
            writer.writerow(["Peak Power", f"{peak_power:.4f}", "W"])
            writer.writerow(["Total Energy", f"{total_energy:.4f}", "J"])
            
        print(f"Logs and Statistics saved to {POWER_LOG_FILE}")
    else:
        print("[ERROR] No power readings collected.")

if __name__ == "__main__":
    run_benchmark()