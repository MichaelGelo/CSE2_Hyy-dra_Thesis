/**
 * @file hyrro_sender.h
 * @brief FPGA board communication via TCP sockets
 * 
 * This file provides functions for:
 * - Direct TCP socket communication with FPGA board
 * - Sending sequences as raw data over TCP
 * - Parsing results from FPGA output
 * - Managing FPGA computation for multiple query-reference pairs
 */

#ifndef HYRRO_SENDER_H
#define HYRRO_SENDER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/time.h>
#include "config.h"
#include "hyrro_results.h"
#include "cpu_utils.h"

// ============================================================================
// TCP COMMUNICATION HELPERS
// ============================================================================

/**
 * @brief Send all bytes through socket
 * @param sock Socket file descriptor
 * @param data Data buffer to send
 * @param size Number of bytes to send
 * @return 0 on success, -1 on failure
 */
static int send_all(int sock, const void *data, size_t size) {
    const char *ptr = (const char *)data;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t sent = send(sock, ptr, remaining, 0);
        if (sent <= 0) return -1;
        ptr += sent;
        remaining -= sent;
    }
    return 0;
}

/**
 * @brief Parse Python list string: "[123, 456, 789]"
 * @param list_str String containing Python list
 * @param res FPGAResult structure to populate
 */
static void parse_python_list(char* list_str, FPGAResult* res) {
    // 1. Remove brackets
    char* start = strchr(list_str, '[');
    char* end = strchr(list_str, ']');
    
    if (!start || !end) return;
    
    *end = '\0'; // Remove closing bracket
    start++;     // Skip opening bracket
    
    // 2. Count items
    int count = 0;
    char* temp = strdup(start);
    char* token = strtok(temp, ", ");
    while(token) {
        if(strlen(token) > 0) count++;
        token = strtok(NULL, ", ");
    }
    free(temp);
    
    res->num_lowest_indexes = count;
    if (count == 0) return;

    // 3. Allocate and fill
    res->lowest_indexes = (unsigned long*)malloc(count * sizeof(unsigned long));
    int i = 0;
    token = strtok(start, ", ");
    while(token) {
        res->lowest_indexes[i++] = strtoul(token, NULL, 10);
        token = strtok(NULL, ", ");
    }
}

/**
 * @brief Read sequence from FASTA file
 * @param filepath Path to FASTA file
 * @param seq_out Output: allocated string with sequence (caller must free)
 * @return Length of sequence, or -1 on error
 */
static int read_fasta_sequence(const char* filepath, char** seq_out) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", filepath);
        return -1;
    }
    
    // Skip header line
    char line[1024];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }
    
    // Read sequence (may be multi-line)
    size_t capacity = 1024;
    size_t length = 0;
    char* sequence = (char*)malloc(capacity);
    
    while (fgets(line, sizeof(line), fp)) {
        // Remove newline
        size_t line_len = strlen(line);
        while (line_len > 0 && (line[line_len-1] == '\n' || line[line_len-1] == '\r')) {
            line[--line_len] = '\0';
        }
        
        // Expand buffer if needed
        if (length + line_len + 1 > capacity) {
            capacity = (length + line_len + 1) * 2;
            sequence = (char*)realloc(sequence, capacity);
        }
        
        // Append line
        strcpy(sequence + length, line);
        length += line_len;
    }
    
    fclose(fp);
    *seq_out = sequence;
    return (int)length;
}

// ============================================================================
// SINGLE FPGA EXECUTION (TCP VERSION)
// ============================================================================

/**
 * @brief Execute FPGA computation for one query-reference pair via TCP
 * 
 * This function:
 * 1. Reads sequences from FASTA files
 * 2. Opens TCP connection to FPGA
 * 3. Sends binary header + raw sequences
 * 4. Receives and parses text response
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

    int sock;
    struct sockaddr_in server_addr;
    struct timeval start, end;

    // ========== Read sequences from FASTA files ==========
    char* ref_seq = NULL;
    char* query_seq = NULL;
    
    int ref_len = read_fasta_sequence(ref_file, &ref_seq);
    int query_len = read_fasta_sequence(query_file, &query_seq);
    
    if (ref_len < 0 || query_len < 0 || !ref_seq || !query_seq) {
        fprintf(stderr, "[FPGA] Failed to read sequences\n");
        if (ref_seq) free(ref_seq);
        if (query_seq) free(query_seq);
        return result;
    }

    // ========== Setup TCP Socket ==========
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        free(ref_seq);
        free(query_seq);
        return result;
    }

    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(FPGA_PORT);
    inet_pton(AF_INET, FPGA_IP, &server_addr.sin_addr);

    // ========== Connect to FPGA ==========
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection to FPGA failed");
        close(sock);
        free(ref_seq);
        free(query_seq);
        return result;
    }

    gettimeofday(&start, NULL);

    // ========== Send Protocol: [QueryLen, RefLen] + Query + Ref ==========
    int header[2] = { query_len, ref_len };
    if (send_all(sock, header, sizeof(header)) != 0) {
        fprintf(stderr, "[FPGA] Failed to send header\n");
        close(sock);
        free(ref_seq);
        free(query_seq);
        return result;
    }

    if (send_all(sock, query_seq, query_len) != 0) {
        fprintf(stderr, "[FPGA] Failed to send query\n");
        close(sock);
        free(ref_seq);
        free(query_seq);
        return result;
    }

    if (send_all(sock, ref_seq, ref_len) != 0) {
        fprintf(stderr, "[FPGA] Failed to send reference\n");
        close(sock);
        free(ref_seq);
        free(query_seq);
        return result;
    }

    // ========== Receive Text Response ==========
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));
    
    ssize_t n = recv(sock, buffer, sizeof(buffer) - 1, 0);
    
    if (n > 0) {
        buffer[n] = '\0'; // Null terminate
        
        // Parse Text Response
        char* line = strtok(buffer, "\n");
        while(line) {
            if (strstr(line, "Hardware Time:")) {
                sscanf(line, "Hardware Time: %lf", &result.hw_exec_time_ms);
            }
            else if (strstr(line, "Final Score:")) {
                sscanf(line, "Final Score: %d", &result.final_score);
            }
            else if (strstr(line, "Lowest Score:")) {
                sscanf(line, "Lowest Score: %d", &result.lowest_score);
            }
            else if (strstr(line, "Indices:")) {
                char* list_start = strchr(line, '[');
                if (list_start) {
                    parse_python_list(list_start, &result);
                }
            }
            line = strtok(NULL, "\n");
        }
    }

    gettimeofday(&end, NULL);
    result.total_exec_ms = get_elapsed_time(start, end) * 1000.0;

    // ========== Cleanup ==========
    close(sock);
    free(ref_seq);
    free(query_seq);

    return result;
}

// ============================================================================
// BATCH FPGA EXECUTION
// ============================================================================

/**
 * @brief Run FPGA computations for multiple queries against one reference via TCP
 * 
 * This function processes all queries against a single reference on the FPGA
 * using direct TCP socket communication.
 * 
 * @param num_queries Number of query sequences
 * @param ref_index Index of reference to process
 * @param total_time_out Output: total execution time in seconds
 * @return Array of FPGAResult structures (one per query)
 * 
 * Process:
 * 1. For each query:
 *    - Open TCP connection
 *    - Send sequences directly
 *    - Receive and parse results
 * 2. Return array of results
 */
static FPGAResult* send_and_run_fpga_single_ref(
    int num_queries, 
    int ref_index, 
    double *total_time_out) 
{
    struct timeval start, end;
    double total_fpga_ms = 0.0;
    
    printf("\n[FPGA] Processing %d queries for Reference %d via TCP\n", 
           num_queries, ref_index);
    
    // Allocate results array
    FPGAResult* results = (FPGAResult*)malloc(num_queries * sizeof(FPGAResult));
    if (!results) {
        fprintf(stderr, "Failed to allocate FPGA results\n");
        if (total_time_out) *total_time_out = 0.0;
        return NULL;
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
        
        // Run single FPGA computation via TCP
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