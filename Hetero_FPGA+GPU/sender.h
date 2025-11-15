#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>   // for basename
#include <regex.h>
#include <sys/time.h>

#define USERNAME        "xilinx"
#define FPGA_IP         "192.168.2.99"
#define REMOTE_PATH     "/home/xilinx/jupyter_notebooks/updatedbit"
#define FPGA_SCRIPT     "fpga_code.py"

#define HOST_FPGA_REF   "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Hetero_FPGA+GPU/fpga_splits/fpga_ref0.fasta"
#define HOST_FPGA_QUERY "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Hetero_FPGA+GPU/fpga_splits/fpga_query0.fasta"

// Path to private SSH key
#define SSH_KEY         "~/.ssh/id_rsa_fpga"

double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

typedef struct {
    double hw_exec_time_ms;
    int final_score;
    int lowest_score;
    unsigned long *lowest_indexes;
    int num_lowest_indexes;
    double overlay_load_ms;
    double total_exec_ms;
} FPGAResult;

void free_fpga_result(FPGAResult *res) {
    if (res->lowest_indexes) free(res->lowest_indexes);
    res->lowest_indexes = NULL;
    res->num_lowest_indexes = 0;
}

FPGAResult send_and_run_fpga() {
    FPGAResult result = {0};
    result.final_score = -1;
    result.lowest_score = -1;
    result.lowest_indexes = NULL;
    result.num_lowest_indexes = 0;

    char command[4096];
    char remote_ref[512], remote_que[512];
    char temp_ref_path[1024], temp_query_path[1024];
    struct timeval start, end;
    double elapsed;

    // Copy string literals to mutable buffers before calling basename
    strncpy(temp_ref_path, HOST_FPGA_REF, sizeof(temp_ref_path) - 1);
    temp_ref_path[sizeof(temp_ref_path) - 1] = '\0';
    strncpy(temp_query_path, HOST_FPGA_QUERY, sizeof(temp_query_path) - 1);
    temp_query_path[sizeof(temp_query_path) - 1] = '\0';

    snprintf(remote_ref, sizeof(remote_ref), "%s/%s", REMOTE_PATH, basename(temp_ref_path));
    snprintf(remote_que, sizeof(remote_que), "%s/%s", REMOTE_PATH, basename(temp_query_path));

    // Send reference file via rsync
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "rsync -avz -e 'ssh -i %s -o StrictHostKeyChecking=no' %s %s@%s:%s/",
             SSH_KEY, HOST_FPGA_REF, USERNAME, FPGA_IP, REMOTE_PATH);
    if (system(command) != 0) fprintf(stderr, "Failed to send reference\n");
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Reference file sent in %.3f seconds.\n\n", elapsed);

    // Send query file via rsync
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "rsync -avz -e 'ssh -i %s -o StrictHostKeyChecking=no' %s %s@%s:%s/",
             SSH_KEY, HOST_FPGA_QUERY, USERNAME, FPGA_IP, REMOTE_PATH);
    if (system(command) != 0) fprintf(stderr, "Failed to send query\n");
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Query file sent in %.3f seconds.\n\n", elapsed);

    // Run FPGA script remotely
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "ssh -i %s -o StrictHostKeyChecking=no %s@%s "
             "\"sudo python3 %s/%s %s %s\"",
             SSH_KEY, USERNAME, FPGA_IP,
             REMOTE_PATH, FPGA_SCRIPT,
             remote_ref, remote_que);

    FILE *fp = popen(command, "r");
    if (!fp) return result;

    char line[4096];
    regex_t re_hw, re_score, re_lowest_score, re_lowest_indexes, re_overlay, re_total;
    regmatch_t m[2];

    // Compile regexes
    regcomp(&re_hw,            "Hardware execution time: ([0-9\\.]+) ms", REG_EXTENDED);
    regcomp(&re_score,         "Final edit distance score: ([0-9]+)", REG_EXTENDED);
    regcomp(&re_lowest_score,  "Lowest score: ([0-9]+)", REG_EXTENDED);
    regcomp(&re_lowest_indexes,"Lowest Score Indexes: \\[(.*)\\]", REG_EXTENDED);
    regcomp(&re_overlay,       "Overlay load time: ([0-9\\.]+) ms", REG_EXTENDED);
    regcomp(&re_total,         "Total script execution time: ([0-9\\.]+) ms", REG_EXTENDED);

    while (fgets(line, sizeof(line), fp)) {
        printf("%s", line);

        if (!regexec(&re_hw, line, 2, m, 0)) {
            char buf[64]; int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len); buf[len] = 0;
            result.hw_exec_time_ms = atof(buf);
        }
        if (!regexec(&re_score, line, 2, m, 0)) {
            char buf[32]; int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len); buf[len] = 0;
            result.final_score = atoi(buf);
        }
        if (!regexec(&re_lowest_score, line, 2, m, 0)) {
            char buf[32]; int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len); buf[len] = 0;
            result.lowest_score = atoi(buf);
        }
        if (!regexec(&re_lowest_indexes, line, 2, m, 0)) {
            char buf[4096]; 
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len); 
            buf[len] = 0;

            char tmp[4096]; strcpy(tmp, buf);
            char *token = strtok(tmp, " ");
            int count = 0;

            // Count valid numbers (not UINT_MAX, not 0)
            while (token) {
                unsigned long val = strtoul(token, NULL, 10);
                if (strcmp(token, "...") != 0 && val != 4294967295UL && val != 0) count++;
                token = strtok(NULL, " ");
            }

            result.num_lowest_indexes = count;
            if (count > 0) {
                result.lowest_indexes = (unsigned long*) malloc(count * sizeof(unsigned long));
                int idx = 0;
                token = strtok(buf, " ");
                while (token) {
                    unsigned long val = strtoul(token, NULL, 10);
                    if (strcmp(token, "...") != 0 && val != 4294967295UL && val != 0)
                        result.lowest_indexes[idx++] = val;
                    token = strtok(NULL, " ");
                }
            }
        }

        if (!regexec(&re_overlay, line, 2, m, 0)) {
            char buf[64]; int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len); buf[len] = 0;
            result.overlay_load_ms = atof(buf);
        }
        if (!regexec(&re_total, line, 2, m, 0)) {
            char buf[64]; int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len); buf[len] = 0;
            result.total_exec_ms = atof(buf);
        }
    }

    regfree(&re_hw);
    regfree(&re_score);
    regfree(&re_lowest_score);
    regfree(&re_lowest_indexes);
    regfree(&re_overlay);
    regfree(&re_total);

    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Remote FPGA script executed and parsed in %.3f seconds.\n\n", elapsed);

    pclose(fp);
    return result;
}
