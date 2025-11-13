#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>   // for basename
#include <sys/time.h> // for gettimeofday

#define USERNAME        "xilinx"
#define PASSWORD        "xilinx"
#define FPGA_IP         "192.168.2.99"
#define REMOTE_PATH     "/home/xilinx/jupyter_notebooks/FPGAGPU1"
#define FPGA_SCRIPT     "fpga_code.py"

// Host PC files
#define HOST_FPGA_REF   "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Final_GPU/fpga_splits/fpga_ref0.fasta"
#define HOST_FPGA_QUERY "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Final_GPU/fpga_splits/fpga_query0.fasta"

double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

void send_and_run_fpga() {
    char command[2048];
    struct timeval start, end;
    double elapsed;

    // ---------------- Send reference file ----------------
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "sshpass -p '%s' rsync '%s' %s@%s:%s/",
             PASSWORD, HOST_FPGA_REF, USERNAME, FPGA_IP, REMOTE_PATH);
    printf("Sending %s to FPGA...\n", HOST_FPGA_REF);
    if (system(command) != 0) {
        fprintf(stderr, "Error sending %s to FPGA.\n", HOST_FPGA_REF);
        return;
    }
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Reference file sent in %.3f seconds.\n\n", elapsed);

    // ---------------- Send query file ----------------
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "sshpass -p '%s' rsync '%s' %s@%s:%s/",
             PASSWORD, HOST_FPGA_QUERY, USERNAME, FPGA_IP, REMOTE_PATH);
    printf("Sending %s to FPGA...\n", HOST_FPGA_QUERY);
    if (system(command) != 0) {
        fprintf(stderr, "Error sending %s to FPGA.\n", HOST_FPGA_QUERY);
        return;
    }
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Query file sent in %.3f seconds.\n\n", elapsed);

    // ---------------- Run FPGA script ----------------
    char remote_ref[512], remote_que[512];
    snprintf(remote_ref, sizeof(remote_ref), "%s/%s", REMOTE_PATH, basename(HOST_FPGA_REF));
    snprintf(remote_que, sizeof(remote_que), "%s/%s", REMOTE_PATH, basename(HOST_FPGA_QUERY));

    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "sshpass -p '%s' ssh %s@%s "
             "\"sudo python3 %s/%s %s %s\"",
             PASSWORD, USERNAME, FPGA_IP,
             REMOTE_PATH, FPGA_SCRIPT, remote_ref, remote_que);
    printf("Executing FPGA script remotely...\n");
    if (system(command) != 0) {
        fprintf(stderr, "Error running FPGA script remotely.\n");
        return;
    }
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("FPGA script executed successfully in %.3f seconds.\n", elapsed);
}
