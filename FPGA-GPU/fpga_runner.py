import paramiko
from scp import SCPClient
from pathlib import Path
import re
import socket

class FPGARunner:
    def __init__(self, hostname="192.168.2.99", username="xilinx", password="xilinx",
                 remote_path="/home/xilinx/jupyter_notebooks/FPGAGPU1", port=22):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.remote_path = remote_path
        self.port = port

        # Establish SSH connection once
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.hostname, username=self.username,
                         password=self.password, port=self.port)
        self.scp = SCPClient(self.ssh.get_transport())

    def send_files(self, local_ref, local_que):
        # Send reference and query to FPGA server
        self.scp.put(local_ref, self.remote_path)
        self.scp.put(local_que, self.remote_path)
        return f"{self.remote_path}/{Path(local_ref).name}", f"{self.remote_path}/{Path(local_que).name}"

    def run_test(self, ref_name, que_name):
        local_ref = Path("./Resources/References") / ref_name
        local_que = Path("./Resources/Queries") / que_name

        if not (local_ref.exists() and local_que.exists()):
            print(f"Skipping: {local_ref} or {local_que} not found")
            return None

        print(f"Sending {ref_name} and {que_name} to FPGA...")

        remote_ref, remote_que = self.send_files(local_ref.as_posix(), local_que.as_posix())

        # Execute the FPGA server script
        cmd = f"python3 {self.remote_path}/fpga_server.py {remote_ref} {remote_que}"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)

        fpga_output = stdout.read().decode()
        fpga_error = stderr.read().decode()
        print(fpga_output)

        # Extract hardware execution time
        match = re.search(r"Hardware execution time: ([\d\.]+) ms", fpga_output)
        hw_exec_time_ms = float(match.group(1)) if match else None

        return hw_exec_time_ms

    def close(self):
        # Close connections when all tests are done
        self.scp.close()
        self.ssh.close()
