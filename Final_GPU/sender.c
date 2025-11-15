#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/time.h>

#define USERNAME        "xilinx"
#define FPGA_IP         "192.168.2.99"
#define REMOTE_PATH     "/home/xilinx/jupyter_notebooks/updatedbit"
#define FPGA_SCRIPT     "fpga_code.py"

#define HOST_FPGA_REF   "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Final_GPU/fpga_splits/fpga_ref0.fasta"
#define HOST_FPGA_QUERY "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Final_GPU/fpga_splits/fpga_query0.fasta"

double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

void send_file_rsync(const char *local, const char *remote_path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "rsync -a --progress '%s' %s@%s:%s/",
             local, USERNAME, FPGA_IP, remote_path);

    printf("Sending %s...\n", local);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to send %s\n", local);
    }
    gettimeofday(&end, NULL);
    printf("%s sent in %.3f seconds.\n\n", basename((char*)local),
           get_elapsed_time(start, end));
}

void run_fpga_script(const char *remote_ref, const char *remote_que) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ssh %s@%s \"sudo python3 %s/%s %s %s\"",
             USERNAME, FPGA_IP, REMOTE_PATH, FPGA_SCRIPT,
             remote_ref, remote_que);

    printf("Running FPGA script...\n");
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if (system(cmd) != 0) {
        fprintf(stderr, "Failed to run FPGA script\n");
    }
    gettimeofday(&end, NULL);
    printf("FPGA script executed in %.3f seconds.\n\n",
           get_elapsed_time(start, end));
}

int main() {
    char remote_ref[512], remote_que[512];
    snprintf(remote_ref, sizeof(remote_ref), "%s/%s", REMOTE_PATH, basename(HOST_FPGA_REF));
    snprintf(remote_que, sizeof(remote_que), "%s/%s", REMOTE_PATH, basename(HOST_FPGA_QUERY));

    send_file_rsync(HOST_FPGA_REF, REMOTE_PATH);
    send_file_rsync(HOST_FPGA_QUERY, REMOTE_PATH);
    run_fpga_script(remote_ref, remote_que);

    return 0;
}