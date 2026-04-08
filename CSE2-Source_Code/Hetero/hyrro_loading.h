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
#include <dirent.h>
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
    
    // If reference is small, let GPU handle it entirely (no FPGA overhead)
    if (total_len < MIN_REF_LENGTH_FOR_FPGA) {
        printf("[SPLIT] Reference too small (%d < %d), GPU-only mode (will partition for parallel processing)\n", 
               total_len, MIN_REF_LENGTH_FOR_FPGA);
        
        // Give entire reference to GPU - it will partition internally
        char *gpu_buf = (char*)malloc((size_t)total_len + 1);
        if (!gpu_buf) {
            fprintf(stderr, "Error: malloc failed for GPU reference\n");
            *gpu_ref_out = NULL;
            *gpu_len_out = 0;
            return;
        }
        memcpy(gpu_buf, sequence, (size_t)total_len);
        gpu_buf[total_len] = '\0';
        *gpu_ref_out = gpu_buf;
        *gpu_len_out = total_len;
        return;  // Skip FPGA split file creation
    }
    
    // Calculate initial GPU portion length based on ratio
    int gpu_len = (int)(total_len * gpu_speed_ratio);
    if (gpu_len > total_len) {
        gpu_len = total_len;
    }

    // Calculate initial FPGA portion length
    int fpga_len = total_len - gpu_len;
    
    // Apply FPGA hardware limit cap
    if (fpga_len > MAX_FPGA_REF_LENGTH) {
        printf("[WARNING] FPGA allocation (%d) exceeds hardware limit (%d)\n", 
               fpga_len, MAX_FPGA_REF_LENGTH);
        printf("[WARNING] Capping FPGA at %d, redistributing overflow to GPU\n", 
               MAX_FPGA_REF_LENGTH);
        
        int overflow = fpga_len - MAX_FPGA_REF_LENGTH;
        fpga_len = MAX_FPGA_REF_LENGTH;
        gpu_len += overflow;
        
        printf("[WARNING] New allocation -> GPU: %d, FPGA: %d\n", gpu_len, fpga_len);
    }

    // Calculate FPGA start with overlap
    // overlap = query_len - 1 (standard for pattern matching)
    int fpga_start = gpu_len - (query_len - 1);
    if (fpga_start < 0) {
        fpga_start = 0;
    }
    if (fpga_start > total_len) {
        fpga_start = total_len;
    }

    printf("[SPLIT] Query length: %d, Overlap: %d\n", query_len, query_len - 1);
    printf("[SPLIT] Reference total: %d, GPU portion: %d, FPGA start: %d (FPGA length: %d)\n", 
           total_len, gpu_len, fpga_start, total_len - fpga_start);

    // Save FPGA portion to file
    save_fpga_split_to_fasta(sequence, fpga_start, ref_no);

    // Allocate and copy GPU portion to memory
    if (gpu_len <= 0) {
        *gpu_ref_out = NULL;
        *gpu_len_out = 0;
        return;
    }

    char *gpu_buf = (char*)malloc((size_t)gpu_len + 1);
    if (!gpu_buf) {
        fprintf(stderr, "Error: malloc failed for GPU reference\n");
        *gpu_ref_out = NULL;
        *gpu_len_out = 0;
        return;
    }
    
    memcpy(gpu_buf, sequence, (size_t)gpu_len);
    gpu_buf[gpu_len] = '\0';
    
    *gpu_ref_out = gpu_buf;
    *gpu_len_out = gpu_len;
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
// FOLDER SCANNING UTILITIES
// ============================================================================

/**
 * @brief Collect all regular-file names from a directory, sorted.
 * @param folder   Directory path to scan.
 * @param count    Output: number of files found.
 * @return Heap-allocated array of heap-allocated filename strings (name only).
 *         Caller must free each element and then the array.
 */
static char** get_files_from_folder(const char* folder, int* count) {
    *count = 0;
    DIR* dir = opendir(folder);
    if (!dir) {
        fprintf(stderr, "Error: cannot open folder: %s\n", folder);
        return NULL;
    }

    // First pass: count regular files
    struct dirent* entry;
    int n = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) n++;
    }

    if (n == 0) {
        closedir(dir);
        return NULL;
    }

    char** files = (char**)malloc(n * sizeof(char*));
    rewinddir(dir);
    int i = 0;
    while ((entry = readdir(dir)) != NULL && i < n) {
        if (entry->d_type == DT_REG) {
            files[i++] = strdup(entry->d_name);
        }
    }
    closedir(dir);

    // Sort alphabetically so ordering is deterministic
    for (int a = 0; a < i - 1; a++) {
        for (int b = a + 1; b < i; b++) {
            if (strcmp(files[a], files[b]) > 0) {
                char* tmp = files[a];
                files[a] = files[b];
                files[b] = tmp;
            }
        }
    }

    *count = i;
    return files;
}

/**
 * @brief Free array returned by get_files_from_folder.
 */
static void free_file_list(char** files, int count) {
    for (int i = 0; i < count; i++) free(files[i]);
    free(files);
}

// ============================================================================
// HIGH-LEVEL LOADING FUNCTIONS (folder-based)
// ============================================================================

/**
 * @brief Load a single query from a FASTA file and save it for FPGA.
 * @param path          Full path to the .fasta file.
 * @param num_queries_out Output: number of sequences loaded (should be 1).
 * @return Array of query strings (caller frees).
 */
static char** load_queries_from_file(const char* path, int* num_queries_out) {
    char** queries = parse_fasta_file(path, num_queries_out);
    if (!queries || *num_queries_out == 0) {
        fprintf(stderr, "Error: Failed to load queries from %s\n", path);
        return NULL;
    }
    printf("Loaded %d quer%s from %s\n",
           *num_queries_out, *num_queries_out == 1 ? "y" : "ies", path);
    save_queries_for_fpga(queries, *num_queries_out);
    return queries;
}

/**
 * @brief Load a single reference from a FASTA file.
 * @param path         Full path to the .fasta file.
 * @param num_refs_out Output: number of sequences loaded (should be 1).
 * @return Array of reference strings (caller frees).
 */
static char** load_references_from_file(const char* path, int* num_refs_out) {
    char** refs = parse_fasta_file(path, num_refs_out);
    if (!refs || *num_refs_out == 0) {
        fprintf(stderr, "Error: Failed to load references from %s\n", path);
        return NULL;
    }
    printf("Loaded %d reference%s from %s\n",
           *num_refs_out, *num_refs_out == 1 ? "" : "s", path);
    return refs;
}

#endif // HYRRO_LOADING_H