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

// Base paths - files will be fpga_ref0.fasta, fpga_ref1.fasta, etc.
#define HOST_FPGA_REF_DIR   "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Hetero_FPGA+GPU/fpga_splits"
#define HOST_FPGA_QUERY_DIR "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Hetero_FPGA+GPU/fpga_splits"

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

// Single FPGA execution (for one query-ref pair)
FPGAResult run_single_fpga(const char* ref_file, const char* query_file) {
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
    strncpy(temp_ref_path, ref_file, sizeof(temp_ref_path) - 1);
    temp_ref_path[sizeof(temp_ref_path) - 1] = '\0';
    strncpy(temp_query_path, query_file, sizeof(temp_query_path) - 1);
    temp_query_path[sizeof(temp_query_path) - 1] = '\0';

    snprintf(remote_ref, sizeof(remote_ref), "%s/%s", REMOTE_PATH, basename(temp_ref_path));
    snprintf(remote_que, sizeof(remote_que), "%s/%s", REMOTE_PATH, basename(temp_query_path));

    // Send reference file via rsync
    printf("Sending reference: %s\n", ref_file);
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "rsync -avz -e 'ssh -i %s -o StrictHostKeyChecking=no' %s %s@%s:%s/",
             SSH_KEY, ref_file, USERNAME, FPGA_IP, REMOTE_PATH);
    if (system(command) != 0) fprintf(stderr, "Failed to send reference\n");
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Reference file sent in %.3f seconds.\n", elapsed);

    // Send query file via rsync
    printf("Sending query: %s\n", query_file);
    gettimeofday(&start, NULL);
    snprintf(command, sizeof(command),
             "rsync -avz -e 'ssh -i %s -o StrictHostKeyChecking=no' %s %s@%s:%s/",
             SSH_KEY, query_file, USERNAME, FPGA_IP, REMOTE_PATH);
    if (system(command) != 0) fprintf(stderr, "Failed to send query\n");
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("Query file sent in %.3f seconds.\n", elapsed);

    // Run FPGA script remotely
    printf("Running FPGA computation...\n");
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

// Run FPGA for multiple queries and references
// Returns 2D array of results: results[query_idx][ref_idx]
FPGAResult** send_and_run_fpga_multi(int num_queries, int num_refs) {
    printf("=== Running FPGA for %d queries x %d references ===\n\n", num_queries, num_refs);
    
    // Allocate 2D array for results
    FPGAResult** results = (FPGAResult**)malloc(num_queries * sizeof(FPGAResult*));
    for (int q = 0; q < num_queries; q++) {
        results[q] = (FPGAResult*)malloc(num_refs * sizeof(FPGAResult));
    }
    
    // Loop through all query-reference pairs
    for (int q = 0; q < num_queries; q++) {
        for (int r = 0; r < num_refs; r++) {
            printf("\n========================================\n");
            printf("Processing: Query %d vs Reference %d\n", q, r);
            printf("========================================\n");
            
            // Build file paths dynamically
            char ref_file[1024];
            char query_file[1024];
            snprintf(ref_file, sizeof(ref_file), "%s/fpga_ref%d.fasta", 
                     HOST_FPGA_REF_DIR, r);
            snprintf(query_file, sizeof(query_file), "%s/fpga_query%d.fasta", 
                     HOST_FPGA_QUERY_DIR, q);
            
            // Run single FPGA computation
            results[q][r] = run_single_fpga(ref_file, query_file);
            
            // Print immediate results
            printf("\n--- Results for Q%d vs R%d in FPGA ---\n", q, r);
            printf("Hardware execution time: %.2f ms\n", results[q][r].hw_exec_time_ms);
            printf("Final score: %d\n", results[q][r].final_score);
            printf("Lowest score: %d\n", results[q][r].lowest_score);
            printf("Lowest indexes count: %d\n", results[q][r].num_lowest_indexes);
            if (results[q][r].num_lowest_indexes > 0) {
                printf("Lowest indexes: [");
                for (int i = 0; i < results[q][r].num_lowest_indexes; i++) {
                    printf("%lu", results[q][r].lowest_indexes[i]);
                    if (i < results[q][r].num_lowest_indexes - 1) printf(", ");
                }
                printf("]\n");
            }
            printf("Total execution time: %.2f ms\n", results[q][r].total_exec_ms);
            printf("----------------------------\n\n");
        }
    }
    
    return results;
}

// Free the 2D results array
void free_fpga_results_multi(FPGAResult** results, int num_queries, int num_refs) {
    if (!results) return;
    
    for (int q = 0; q < num_queries; q++) {
        for (int r = 0; r < num_refs; r++) {
            free_fpga_result(&results[q][r]);
        }
        free(results[q]);
    }
    free(results);
}

// Legacy single-pair function (for backward compatibility)
FPGAResult send_and_run_fpga() {
    char ref_file[1024];
    char query_file[1024];
    snprintf(ref_file, sizeof(ref_file), "%s/fpga_ref0.fasta", HOST_FPGA_REF_DIR);
    snprintf(query_file, sizeof(query_file), "%s/fpga_query0.fasta", HOST_FPGA_QUERY_DIR);
    
    return run_single_fpga(ref_file, query_file);
}