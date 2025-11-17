#ifndef LOADING_H
#define LOADING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "C_utils.h"

// ================= USER DEFINES =================
#define QUERY_FILE     "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/Queries/multique2_128.fasta"
#define REFERENCE_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/References/multipleref1_1M.fasta"
#define FPGA_OUTPUT_DIR "./fpga_splits/"

// Speed ratio for splitting reference
#define FPGA_SPEED_RATIO 0.5f
#define GPU_SPEED_RATIO  0.5f
// ===============================================

typedef struct {
    double prev_gpu_time;
    double prev_fpga_time;
    float new_gpu_ratio;
} RatioThreadArgs;


static float calculate_gpu_ratio(double gpu_time, double fpga_time) {
    if (gpu_time <= 0 || fpga_time <= 0) {
        return GPU_SPEED_RATIO; // fallback to default
    }
    
    // The faster device should get MORE work
    // Ratio based on inverse of time (throughput)
    double gpu_throughput = 1.0 / gpu_time;
    double fpga_throughput = 1.0 / fpga_time;
    double total_throughput = gpu_throughput + fpga_throughput;
    
    float new_gpu_ratio = (float)(gpu_throughput / total_throughput);
    
    // Clamp to reasonable bounds

    
    return new_gpu_ratio;
}

// Ratio calculation thread function
void* ratio_thread_func(void* arg) {
    RatioThreadArgs* args = (RatioThreadArgs*)arg;

    // Calculate new GPU ratio
    args->new_gpu_ratio = calculate_gpu_ratio(args->prev_gpu_time, args->prev_fpga_time);

    // Print the computed ratios
    printf("[Ratio Thread] Computed speed ratios: GPU = %.3f, FPGA = %.3f\n",
           args->new_gpu_ratio, 1.0f - args->new_gpu_ratio);

    return NULL;
}

// Utility: write sequence to FASTA (line width 60)
static void write_fasta(FILE *file, const char *sequence) {
    int len = (int)strlen(sequence);
    for (int i = 0; i < len; i += 60) {
        fprintf(file, "%.*s\n", (len - i >= 60 ? 60 : len - i), sequence + i);
    }
}

// Save FPGA split of reference to FASTA
static void save_fpga_split_to_fasta(const char *sequence, int start_index, int ref_no) {
    int len = (int)strlen(sequence) - start_index;
    if (len <= 0) return;

    // ensure directory exists
    struct stat st = {0};
    if (stat(FPGA_OUTPUT_DIR, &st) == -1) mkdir(FPGA_OUTPUT_DIR, 0755);

    char filename[512];
    snprintf(filename, sizeof(filename), "%sfpga_ref%d.fasta", FPGA_OUTPUT_DIR, ref_no);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return;
    }

    fprintf(fp, ">FPGA_Ref_%d\n", ref_no);
    for (int i = 0; i < len; i++) {
        fputc(sequence[start_index + i], fp);
        if ((i + 1) % 60 == 0) fputc('\n', fp);
    }
    if (len % 60 != 0) fputc('\n', fp);
    fclose(fp);

    printf("FPGA split saved: %s (start=%d, length=%d)\n", filename, start_index, len);
}

// Split reference into GPU in-memory and FPGA FASTA
static void split_reference_for_fpga_gpu(const char *sequence, int query_len,
                                        char **gpu_ref_out, int *gpu_len_out, int ref_no, float gpu_speed_ratio)
{
    int total_len = (int)strlen(sequence);
    int gpu_len = (int)(total_len * gpu_speed_ratio);
    if (gpu_len > total_len) gpu_len = total_len;

    int fpga_start = gpu_len - (query_len - 1);
    if (fpga_start < 0) fpga_start = 0;
    if (fpga_start > total_len) fpga_start = total_len;

    // Save FPGA portion
    save_fpga_split_to_fasta(sequence, fpga_start, ref_no);

    // GPU portion
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

// Save queries for FPGA to FASTA
static void save_queries_for_fpga(char **queries, int num_queries) {
    struct stat st = {0};
    if (stat(FPGA_OUTPUT_DIR, &st) == -1) mkdir(FPGA_OUTPUT_DIR, 0755);

    for (int i = 0; i < num_queries; ++i) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%sfpga_query%d.fasta", FPGA_OUTPUT_DIR, i);
        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error: could not open %s for writing\n", filename);
            continue;
        }
        fprintf(fp, ">FPGA_Query_%d\n", i);
        write_fasta(fp, queries[i]);
        fclose(fp);
        printf("FPGA query saved: %s (length=%zu)\n", filename, strlen(queries[i]));
    }
}

// Load queries into memory and optionally save for FPGA
static char** load_queries(int *num_queries_out) {
    char **queries = parse_fasta_file(QUERY_FILE, num_queries_out);
    if (queries && *num_queries_out > 0) save_queries_for_fpga(queries, *num_queries_out);
    return queries;
}

// Load references, split GPU/FPGA, store GPU in-memory
static char** load_references_gpu_fpga(int *num_refs_out, char ***gpu_refs_out, int **gpu_lens_out, int query_len) {
    int num_refs = 0;
    

    char **refs = parse_fasta_file(REFERENCE_FILE, &num_refs);
    if (!refs || num_refs <= 0) return NULL;

    char **gpu_refs = (char**)malloc(sizeof(char*) * num_refs);
    int *gpu_lens = (int*)malloc(sizeof(int) * num_refs);
    if (!gpu_refs || !gpu_lens) {
        fprintf(stderr, "OOM for GPU references\n");
        return NULL;
    }

    for (int i = 0; i < num_refs; ++i) {  //dito ung dynamic and ung threading
        //new speed ratio
        //split_reference_for_fpga_gpu(refs[i], query_len, &gpu_refs[i], &gpu_lens[i], i);
        //get speed ratio
    }

    *gpu_refs_out = gpu_refs;
    *gpu_lens_out = gpu_lens;
    *num_refs_out = num_refs;
    return refs;
}
  

#endif // LOADING_H