/**
 * @file hyrro_sender.h
 * @brief FPGA board communication and result parsing
 * 
 * This file provides functions for:
 * - Sending files to FPGA board via SSH/rsync
 * - Executing Python scripts remotely on FPGA
 * - Parsing results from FPGA output
 * - Managing FPGA computation for multiple query-reference pairs
 */

#ifndef HYRRO_SENDER_H
#define HYRRO_SENDER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <regex.h>
#include <sys/time.h>
#include "config.h"
#include "hyrro_results.h"
#include "cpu_utils.h"

// ============================================================================
// SINGLE FPGA EXECUTION
// ============================================================================

/**
 * @brief Execute FPGA computation for one query-reference pair
 * 
 * This function:
 * 1. Sends reference file to FPGA via rsync
 * 2. Executes Python script on FPGA remotely via SSH
 * 3. Parses output to extract results
 * 
 * @param ref_file Path to reference FASTA file
 * @param query_file Path to query FASTA file
 * @return FPGAResult structure with timing and distance information
 */
static FPGAResult run_single_fpga(const char* ref_file, const char* query_file) {
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

    // Copy paths to mutable buffers (basename modifies its argument)
    strncpy(temp_ref_path, ref_file, sizeof(temp_ref_path) - 1);
    temp_ref_path[sizeof(temp_ref_path) - 1] = '\0';
    strncpy(temp_query_path, query_file, sizeof(temp_query_path) - 1);
    temp_query_path[sizeof(temp_query_path) - 1] = '\0';

    // Build remote file paths
    snprintf(remote_ref, sizeof(remote_ref), "%s/%s", REMOTE_PATH, basename(temp_ref_path));
    snprintf(remote_que, sizeof(remote_que), "%s/%s", REMOTE_PATH, basename(temp_query_path));

    // ========== Send reference file via rsync ==========
    printf("[FPGA] Sending reference...\n");
    gettimeofday(&start, NULL);
    
    snprintf(command, sizeof(command),
             "rsync -avz -e 'ssh -i %s -o StrictHostKeyChecking=no' %s %s@%s:%s/",
             SSH_KEY, ref_file, USERNAME, FPGA_IP, REMOTE_PATH);
    
    if (system(command) != 0) {
        fprintf(stderr, "Failed to send reference\n");
    }
    
    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("[FPGA] Reference sent (%.2fs)\n", elapsed);

    // ========== Run FPGA script remotely ==========
    printf("[FPGA] Running computation...\n");
    gettimeofday(&start, NULL);
    
    snprintf(command, sizeof(command),
             "ssh -i %s -o StrictHostKeyChecking=no %s@%s "
             "\"sudo python3 %s/%s %s %s\"",
             SSH_KEY, USERNAME, FPGA_IP,
             REMOTE_PATH, FPGA_SCRIPT,
             remote_ref, remote_que);

    FILE *fp = popen(command, "r");
    if (!fp) {
        fprintf(stderr, "Failed to execute FPGA script\n");
        return result;
    }

    // ========== Parse output using regex ==========
    char line[4096];
    regex_t re_hw, re_score, re_lowest_score, re_lowest_indexes, re_overlay, re_total;
    regmatch_t m[2];

    // Compile regular expressions for parsing
    regcomp(&re_hw, "Hardware execution time: ([0-9\\.]+) ms", REG_EXTENDED);
    regcomp(&re_score, "Final edit distance score: ([0-9]+)", REG_EXTENDED);
    regcomp(&re_lowest_score, "Lowest score: ([0-9]+)", REG_EXTENDED);
    regcomp(&re_lowest_indexes, "Lowest Score Indexes: \\[(.*)\\]", REG_EXTENDED);
    regcomp(&re_overlay, "Overlay load time: ([0-9\\.]+) ms", REG_EXTENDED);
    regcomp(&re_total, "Total script execution time: ([0-9\\.]+) ms", REG_EXTENDED);

    // Read and parse each line of output
    while (fgets(line, sizeof(line), fp)) {
        // printf("%s", line);  // Suppress verbose FPGA output

        // Parse hardware execution time
        if (!regexec(&re_hw, line, 2, m, 0)) {
            char buf[64];
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len);
            buf[len] = 0;
            result.hw_exec_time_ms = atof(buf);
        }
        
        // Parse final score
        if (!regexec(&re_score, line, 2, m, 0)) {
            char buf[32];
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len);
            buf[len] = 0;
            result.final_score = atoi(buf);
        }
        
        // Parse lowest score
        if (!regexec(&re_lowest_score, line, 2, m, 0)) {
            char buf[32];
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len);
            buf[len] = 0;
            result.lowest_score = atoi(buf);
        }
        
        // Parse lowest score indexes
        if (!regexec(&re_lowest_indexes, line, 2, m, 0)) {
            char buf[4096];
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len);
            buf[len] = 0;

            // First pass: count valid numbers
            char tmp[4096];
            strcpy(tmp, buf);
            char *token = strtok(tmp, " ");
            int count = 0;

            while (token) {
                unsigned long val = strtoul(token, NULL, 10);
                // Skip invalid markers (UINT_MAX, 0, "...")
                if (strcmp(token, "...") != 0 && val != 4294967295UL && val != 0) {
                    count++;
                }
                token = strtok(NULL, " ");
            }

            result.num_lowest_indexes = count;
            
            // Second pass: extract valid numbers
            if (count > 0) {
                result.lowest_indexes = (unsigned long*)malloc(count * sizeof(unsigned long));
                int idx = 0;
                token = strtok(buf, " ");
                
                while (token) {
                    unsigned long val = strtoul(token, NULL, 10);
                    if (strcmp(token, "...") != 0 && val != 4294967295UL && val != 0) {
                        result.lowest_indexes[idx++] = val;
                    }
                    token = strtok(NULL, " ");
                }
            }
        }

        // Parse overlay load time
        if (!regexec(&re_overlay, line, 2, m, 0)) {
            char buf[64];
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len);
            buf[len] = 0;
            result.overlay_load_ms = atof(buf);
        }
        
        // Parse total execution time
        if (!regexec(&re_total, line, 2, m, 0)) {
            char buf[64];
            int len = m[1].rm_eo - m[1].rm_so;
            strncpy(buf, line + m[1].rm_so, len);
            buf[len] = 0;
            result.total_exec_ms = atof(buf);
        }
    }

    // Clean up regex objects
    regfree(&re_hw);
    regfree(&re_score);
    regfree(&re_lowest_score);
    regfree(&re_lowest_indexes);
    regfree(&re_overlay);
    regfree(&re_total);

    gettimeofday(&end, NULL);
    elapsed = get_elapsed_time(start, end);
    printf("[FPGA] Computation complete (%.2fs)\n", elapsed);

    pclose(fp);
    return result;
}

// ============================================================================
// BATCH FPGA EXECUTION
// ============================================================================

/**
 * @brief Run FPGA computations for multiple queries against one reference
 * 
 * This function processes all queries against a single reference on the FPGA.
 * It's designed to minimize file transfer overhead by sending queries once
 * and reusing them for multiple references.
 * 
 * @param num_queries Number of query sequences
 * @param ref_index Index of reference to process
 * @param total_time_out Output: total execution time in seconds
 * @return Array of FPGAResult structures (one per query)
 * 
 * Process:
 * 1. Send all queries to FPGA (if not already there)
 * 2. For each query:
 *    - Run FPGA computation against specified reference
 *    - Parse and store results
 * 3. Return array of results
 */
static FPGAResult* send_and_run_fpga_single_ref(
    int num_queries, 
    int ref_index, 
    double *total_time_out) 
{
    char command[4096];
    struct timeval start, end;
    double total_fpga_ms = 0.0;
    
    printf("\n[FPGA] Processing %d queries for Reference %d\n", 
           num_queries, ref_index);
    
    // Allocate results array
    FPGAResult* results = (FPGAResult*)malloc(num_queries * sizeof(FPGAResult));
    if (!results) {
        fprintf(stderr, "Failed to allocate FPGA results\n");
        if (total_time_out) *total_time_out = 0.0;
        return NULL;
    }
    
    // ========== Send all queries first (if needed) ==========
    for (int q = 0; q < num_queries; q++) {
        char query_file[1024];
        
        snprintf(query_file, sizeof(query_file), "%s/fpga_query%d.fasta", 
                 HOST_FPGA_QUERY_DIR, q);
        // printf("[FPGA] Sending query %d...\n", q);  // Suppress per-query messages
        
        snprintf(command, sizeof(command),
                 "rsync -avz -e 'ssh -i %s -o StrictHostKeyChecking=no' %s %s@%s:%s/",
                 SSH_KEY, query_file, USERNAME, FPGA_IP, REMOTE_PATH);
        
        if (system(command) != 0) {
            fprintf(stderr, "Failed to send query\n");
        }
    }
    
    // Build reference file path
    char ref_file[1024];
    snprintf(ref_file, sizeof(ref_file), "%s/fpga_ref%d.fasta", 
             HOST_FPGA_REF_DIR, ref_index);
    
    // ========== Process each query against this reference ==========
    for (int q = 0; q < num_queries; q++) {
        char query_file[1024];
        snprintf(query_file, sizeof(query_file), "%s/fpga_query%d.fasta", 
                 HOST_FPGA_QUERY_DIR, q);
        
        // Run single FPGA computation
        results[q] = run_single_fpga(ref_file, query_file);
        
        // Compact results output
        printf("[FPGA] Q%d: Score=%d, Lowest=%d (count=%d), Time=%.1fms\n", 
               q, results[q].final_score, results[q].lowest_score,
               results[q].num_lowest_indexes, results[q].hw_exec_time_ms);
        
        // Accumulate time
        total_fpga_ms += results[q].hw_exec_time_ms;
    }
    
    // Convert to seconds
    double total_fpga_sec = total_fpga_ms / 1000.0;

    if (total_time_out) {
        *total_time_out = total_fpga_sec;
    }
    
    printf("=== FPGA Total Time for Ref %d: %.6f seconds ===\n", 
           ref_index, total_fpga_sec);
    
    return results;
}

// ============================================================================
// LEGACY COMPATIBILITY FUNCTION
// ============================================================================

/**
 * @brief Legacy single-pair FPGA function (for backward compatibility)
 * @return FPGAResult for first query-reference pair
 * 
 * Processes fpga_query0.fasta vs fpga_ref0.fasta
 */
static FPGAResult send_and_run_fpga() {
    char ref_file[1024];
    char query_file[1024];
    snprintf(ref_file, sizeof(ref_file), "%s/fpga_ref0.fasta", HOST_FPGA_REF_DIR);
    snprintf(query_file, sizeof(query_file), "%s/fpga_query0.fasta", HOST_FPGA_QUERY_DIR);
    
    return run_single_fpga(ref_file, query_file);
}

#endif // HYRRO_SENDER_H