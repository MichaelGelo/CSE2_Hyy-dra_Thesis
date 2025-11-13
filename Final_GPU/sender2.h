#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

#define USERNAME        "xilinx"
#define PASSWORD        "xilinx"
#define FPGA_IP         "192.168.2.99"
#define REMOTE_PATH     "/home/xilinx/jupyter_notebooks/FPGAGPU1"
#define FPGA_SCRIPT     "fpga_code.py"

#define HOST_FPGA_REF   "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Final_GPU/fpga_splits/fpga_ref0.fasta"
#define HOST_FPGA_QUERY "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Final_GPU/fpga_splits/fpga_query0.fasta"

typedef struct {
    double hw_exec_time_ms;
    int final_score;
    int zero_count;
    int zero_indices[1024];
} FPGAResult;

void send_and_run_fpga(FPGAResult* res) {
    char command[2048];
    char output_file[] = "/tmp/fpga_output.txt";

    // Send reference
    snprintf(command, sizeof(command),
             "sshpass -p '%s' scp '%s' %s@%s:%s/",
             PASSWORD, HOST_FPGA_REF, USERNAME, FPGA_IP, REMOTE_PATH);
    system(command);

    // Send query
    snprintf(command, sizeof(command),
             "sshpass -p '%s' scp '%s' %s@%s:%s/",
             PASSWORD, HOST_FPGA_QUERY, USERNAME, FPGA_IP, REMOTE_PATH);
    system(command);

    // Run FPGA script and capture output
    snprintf(command, sizeof(command),
             "sshpass -p '%s' ssh %s@%s \"sudo python3 %s/%s %s %s\"",
             PASSWORD, USERNAME, FPGA_IP,
             REMOTE_PATH, FPGA_SCRIPT,
             basename(HOST_FPGA_REF), basename(HOST_FPGA_QUERY));

    FILE* fp = popen(command, "r");
    if (!fp) {
        fprintf(stderr, "Error running FPGA script\n");
        return;
    }

    char line[1024];
    res->hw_exec_time_ms = 0.0;
    res->final_score = -1;
    res->zero_count = 0;
    memset(res->zero_indices, -1, sizeof(res->zero_indices));

    while (fgets(line, sizeof(line), fp)) {
        double time_val;
        int score_val;

        if (sscanf(line, "Hardware execution time: %lf ms", &time_val) == 1) {
            res->hw_exec_time_ms = time_val;
        } else if (sscanf(line, "Final edit distance score: %d", &score_val) == 1) {
            res->final_score = score_val;
        } else if (strstr(line, "Zero Indexes") != NULL) {
            // Parse the indexes
            char* start = strchr(line, '[');
            char* end = strchr(line, ']');
            if (start && end && end > start) {
                start++;
                char* token = strtok(start, " ,");
                while (token && res->zero_count < 1024) {
                    int val = atoi(token);
                    // Only keep valid indexes (discard 0 and 4294967295)
                    if (val != 0 && val != -1 && val != 4294967295U) {
                        res->zero_indices[res->zero_count++] = val;
                    }
                    token = strtok(NULL, " ,");
                }
            }
        }
    }

    pclose(fp);
}
