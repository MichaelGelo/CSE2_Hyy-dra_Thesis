#ifndef LOADING_H
#define LOADING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>
#include <math.h>
#include "C_utils.h"

// ================= USER DEFINES =================
#define QUERY_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/Queries/mque1_256.fasta"
#define REFERENCE_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/References/mref_8_50M.fasta"
#define FPGA_OUTPUT_DIR "./fpga_splits/"

// Speed ratio defaults
#define FPGA_SPEED_RATIO 0.5f
#define GPU_SPEED_RATIO  0.5f
// ===============================================

// ============= SYNCHRONIZATION PRIMITIVES ============
static pthread_mutex_t ratio_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  ratio_cond  = PTHREAD_COND_INITIALIZER;
// =====================================================

// RatioThreadArgs: pass old ratio in, receive new ratio out
typedef struct {
    double prev_gpu_time;
    double prev_fpga_time;
    float  old_gpu_ratio;   // PASS THE CURRENT gpu_ratio IN HERE
    float  new_gpu_ratio;   // UPDATED RATIO WRITTEN BACK HERE
    int    taken;           // handshake flag between caller and ratio thread
} RatioThreadArgs;

// Stable scheduler parameters
#define MIN_GPU_RATIO 0.05
#define MAX_GPU_RATIO 0.95
#define RATIO_SMOOTHING_ALPHA 0.5   // 0..1, higher = faster adaptation, lower = smoother

// calculate_gpu_ratio: stable multiplicative correction using old_ratio
static float calculate_gpu_ratio(double gpu_time, double fpga_time, float old_ratio) {
    if (old_ratio <= 0.0f) old_ratio = GPU_SPEED_RATIO;
    if (gpu_time <= 0.0 || fpga_time <= 0.0) {
        return old_ratio;
    }

    // multiplicative correction to drive T_gpu == T_fpga:
    // new = old * sqrt(fpga_time / gpu_time)
    double factor = sqrt(fpga_time / gpu_time);
    double proposed = (double)old_ratio * factor;

    // smoothing
    double blended = (1.0 - RATIO_SMOOTHING_ALPHA) * old_ratio + RATIO_SMOOTHING_ALPHA * proposed;

    // clamp
    if (blended < MIN_GPU_RATIO) blended = MIN_GPU_RATIO;
    if (blended > MAX_GPU_RATIO) blended = MAX_GPU_RATIO;

    return (float)blended;
}

// ratio_thread_func: uses only data passed in args, does not touch any global ratio_state
void* ratio_thread_func(void* arg) {
    RatioThreadArgs* args = (RatioThreadArgs*)arg;

    // snapshot shared values under mutex handshake
    pthread_mutex_lock(&ratio_mutex);

    double gpu_t  = args->prev_gpu_time;
    double fpga_t = args->prev_fpga_time;
    float  old_r  = args->old_gpu_ratio;

    args->taken = 1;
    pthread_cond_signal(&ratio_cond);
    pthread_mutex_unlock(&ratio_mutex);

    // compute new ratio using the stable rule
    args->new_gpu_ratio = calculate_gpu_ratio(gpu_t, fpga_t, old_r);

    // diagnostics, useful for debugging
    printf("[Ratio Thread] Prev times: GPU = %.6f s, FPGA = %.6f s\n", gpu_t, fpga_t);
    printf("[Ratio Thread] Old ratio: GPU = %.3f FPGA = %.3f\n", old_r, 1.0f - old_r);
    printf("[Ratio Thread] New ratio: GPU = %.3f FPGA = %.3f\n",
           args->new_gpu_ratio, 1.0f - args->new_gpu_ratio);

    return NULL;
}

// ==================== File IO / split helpers (unchanged but included) ===================

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

#endif // LOADING_H
