/**
 * @file hyrro_loading.h
 * @brief File I/O and reference splitting utilities
 * 
 * This file provides functions for:
 * - Loading queries and references from FASTA files
 * - Splitting references for heterogeneous GPU-FPGA processing
 * - Saving FPGA portions to separate files
 */

#ifndef HYRRO_LOADING_H
#define HYRRO_LOADING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include "config.h"
#include "cpu_utils.h"

// ============================================================================
// FASTA FILE UTILITIES
// ============================================================================

/**
 * @brief Write sequence to FASTA file with line wrapping
 * @param file File pointer to write to
 * @param sequence Sequence string to write
 * 
 * Writes sequence with 60 characters per line (standard FASTA format).
 */
static void write_fasta(FILE *file, const char *sequence) {
    int len = (int)strlen(sequence);
    
    // Write in lines of 60 characters
    for (int i = 0; i < len; i += 60) {
        int line_len = (len - i >= 60) ? 60 : (len - i);
        fprintf(file, "%.*s\n", line_len, sequence + i);
    }
}

// ============================================================================
// REFERENCE SPLITTING FOR HETEROGENEOUS PROCESSING
// ============================================================================

/**
 * @brief Save FPGA portion of reference to FASTA file
 * @param sequence Full reference sequence
 * @param start_index Starting position for FPGA portion
 * @param ref_no Reference number (for filename)
 * 
 * Creates: fpga_splits/fpga_ref{ref_no}.fasta
 * 
 * The FPGA portion starts at start_index and extends to the end of the
 * reference. This overlaps with the GPU portion to ensure no matches are
 * missed at the boundary.
 */
static void save_fpga_split_to_fasta(const char *sequence, int start_index, int ref_no) {
    int len = (int)strlen(sequence) - start_index;
    if (len <= 0) return;

    // Ensure output directory exists
    struct stat st = {0};
    if (stat(FPGA_OUTPUT_DIR, &st) == -1) {
        mkdir(FPGA_OUTPUT_DIR, 0755);
    }

    // Create filename: fpga_splits/fpga_ref0.fasta, etc.
    char filename[512];
    snprintf(filename, sizeof(filename), "%sfpga_ref%d.fasta", FPGA_OUTPUT_DIR, ref_no);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return;
    }

    // Write header
    fprintf(fp, ">FPGA_Ref_%d\n", ref_no);
    
    // Write sequence with line wrapping
    for (int i = 0; i < len; i++) {
        fputc(sequence[start_index + i], fp);
        if ((i + 1) % 60 == 0) {
            fputc('\n', fp);
        }
    }
    
    // Add final newline if needed
    if (len % 60 != 0) {
        fputc('\n', fp);
    }
    
    fclose(fp);

    printf("FPGA split saved: %s (start=%d, length=%d)\n", filename, start_index, len);
}

/**
 * @brief Split reference for GPU and FPGA processing
 * 
 * Splits reference into two parts based on gpu_speed_ratio:
 * - GPU portion: first gpu_ratio * total_length characters (in memory)
 * - FPGA portion: remaining characters with overlap (saved to file)
 * 
 * The overlap ensures that matches spanning the boundary are not missed.
 * 
 * @param sequence Full reference sequence
 * @param query_len Length of query (determines overlap size)
 * @param gpu_ref_out Output: GPU portion (allocated string)
 * @param gpu_len_out Output: length of GPU portion
 * @param ref_no Reference number (for FPGA filename)
 * @param gpu_speed_ratio Fraction of reference for GPU (0.0 to 1.0)
 * 
 * Example:
 *   Reference: "ABCDEFGHIJKLMNOP" (16 chars)
 *   gpu_speed_ratio = 0.5, query_len = 4
 *   
 *   GPU portion:  "ABCDEFGH" (8 chars)
 *   FPGA portion: "EFGHIJKLMNOP" (starts at position 5 = 8 - (4-1))
 *   
 *   Overlap: "EFGH" ensures matches spanning position 8 are found
 */
static void split_reference_for_fpga_gpu(
    const char *sequence, 
    int query_len,
    char **gpu_ref_out, 
    int *gpu_len_out, 
    int ref_no, 
    float gpu_speed_ratio)
{
    int total_len = (int)strlen(sequence);
    
    // Calculate GPU portion length
    int gpu_len = (int)(total_len * gpu_speed_ratio);
    if (gpu_len > total_len) {
        gpu_len = total_len;
    }

    // CRITICAL FIX: FPGA must always start from position 0!
    // The FPGA algorithm has position-dependent behavior and gives incorrect
    // results when starting from arbitrary positions. To work around this,
    // we always give FPGA the FULL reference starting from position 0.
    int fpga_start = 0;

    // Save FPGA portion to file (ALWAYS from position 0)
    save_fpga_split_to_fasta(sequence, fpga_start, ref_no);

    // Allocate and copy GPU portion to memory
    // GPU needs extended reference for sliding window computation
    if (gpu_len <= 0) {
        *gpu_ref_out = NULL;
        *gpu_len_out = 0;
        return;
    }

    // Extend GPU buffer to include characters needed for final sliding windows
    int gpu_extended_len = gpu_len + (query_len - 1);
    if (gpu_extended_len > total_len) {
        gpu_extended_len = total_len;
    }

    char *gpu_buf = (char*)malloc((size_t)gpu_extended_len + 1);
    if (!gpu_buf) {
        fprintf(stderr, "Error: malloc failed for GPU reference\n");
        *gpu_ref_out = NULL;
        *gpu_len_out = 0;
        return;
    }
    
    memcpy(gpu_buf, sequence, (size_t)gpu_extended_len);
    gpu_buf[gpu_extended_len] = '\0';
    
    *gpu_ref_out = gpu_buf;
    *gpu_len_out = gpu_extended_len;  // Report extended length
}

// ============================================================================
// QUERY SAVING FOR FPGA
// ============================================================================

/**
 * @brief Save queries to FASTA files for FPGA processing
 * @param queries Array of query strings
 * @param num_queries Number of queries
 * 
 * Creates: fpga_splits/fpga_query0.fasta, fpga_query1.fasta, etc.
 * 
 * Each query is saved as a separate FASTA file that will be sent to
 * the FPGA board for processing.
 */
static void save_queries_for_fpga(char **queries, int num_queries) {
    // Ensure directory exists
    struct stat st = {0};
    if (stat(FPGA_OUTPUT_DIR, &st) == -1) {
        mkdir(FPGA_OUTPUT_DIR, 0755);
    }

    for (int i = 0; i < num_queries; ++i) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%sfpga_query%d.fasta", 
                 FPGA_OUTPUT_DIR, i);
        
        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error: could not open %s for writing\n", filename);
            continue;
        }
        
        // Write header and sequence
        fprintf(fp, ">FPGA_Query_%d\n", i);
        write_fasta(fp, queries[i]);
        
        fclose(fp);
        
        printf("FPGA query saved: %s (length=%zu)\n", filename, strlen(queries[i]));
    }
}

// ============================================================================
// HIGH-LEVEL LOADING FUNCTIONS
// ============================================================================

/**
 * @brief Load queries from FASTA file and save for FPGA
 * @param num_queries_out Output: number of queries loaded
 * @return Array of query strings
 * 
 * Loads queries from QUERY_FILE (defined in config.h) and automatically
 * saves copies for FPGA processing.
 */
static char** load_queries(int *num_queries_out) {
    // Parse FASTA file
    char **queries = parse_fasta_file(QUERY_FILE, num_queries_out);
    
    if (!queries || *num_queries_out == 0) {
        fprintf(stderr, "Error: Failed to load queries from %s\n", QUERY_FILE);
        return NULL;
    }
    
    printf("Loaded %d queries from %s\n", *num_queries_out, QUERY_FILE);
    
    // Save copies for FPGA
    save_queries_for_fpga(queries, *num_queries_out);
    
    return queries;
}

/**
 * @brief Load references from FASTA file
 * @param num_refs_out Output: number of references loaded
 * @return Array of reference strings
 * 
 * Loads references from REFERENCE_FILE (defined in config.h).
 */
static char** load_references(int *num_refs_out) {
    char **refs = parse_fasta_file(REFERENCE_FILE, num_refs_out);
    
    if (!refs || *num_refs_out == 0) {
        fprintf(stderr, "Error: Failed to load references from %s\n", REFERENCE_FILE);
        return NULL;
    }
    
    printf("Loaded %d references from %s\n", *num_refs_out, REFERENCE_FILE);
    
    return refs;
}

#endif // HYRRO_LOADING_H