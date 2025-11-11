from pathlib import Path
import paramiko
from scp import SCPClient
import re

def run_fpga_test(ref_name, que_name,
                  hostname="192.168.2.99",
                  username="xilinx",
                  password="xilinx",
                  remote_path="/home/xilinx/jupyter_notebooks/FPGAGPU1",
                  main_script="fpga_code.py"):
    """
    Sends a reference and query FASTA file to the FPGA,
    runs the processing script, and returns the execution time (ms).
    """

    # local base paths
    ref_base = Path("")
    que_base = Path("")

    local_ref = (ref_base / ref_name).resolve()
    local_que = (que_base / que_name).resolve()

    if not (local_ref.exists()):
        print(f"Skipping: {local_ref.as_posix()}not found")
        return None
    
    if not (local_que.exists()):
        print(f"Skipping: {local_que.as_posix()} not found")
        return None

    print(f"Sending {ref_name} and {que_name} to FPGA...")

    # ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)
    scp = SCPClient(ssh.get_transport())

    # send files
    scp.put(local_ref.as_posix(), remote_path)
    scp.put(local_que.as_posix(), remote_path)

    # build remote file paths
    remote_ref = f"{remote_path}/{ref_name}"
    remote_que = f"{remote_path}/{que_name}"

    cmd = f"sudo python3 {remote_path}/{main_script} {remote_ref} {remote_que}"
    print(f"Executing FPGA command")

    stdin, stdout, stderr = ssh.exec_command(cmd)
    fpga_output = stdout.read().decode()
    fpga_error = stderr.read().decode()

    # display and extract results

    print(fpga_output)

    match = re.search(r"Hardware execution time: ([\d\.]+) ms", fpga_output)
    hw_exec_time_ms = float(match.group(1)) if match else None

    # close connections
    scp.close()
    ssh.close()

    return hw_exec_time_ms