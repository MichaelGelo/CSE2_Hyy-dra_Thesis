/**
 * @file finalcuda.cu
 * @brief Main program for heterogeneous GPU-FPGA Damerau-Levenshtein distance computation
 * 
 * This program implements a heterogeneous computing approach that dynamically
 * distributes workload between GPU and FPGA based on their relative performance.
 * 
 * Architecture:
 * - GPU: Processes first portion of each reference using CUDA
 * - FPGA: Processes remaining portion via SSH/rsync
 * - Adaptive scheduler adjusts work distribution based on execution times
 * 
 * Algorithm: Hyyrö variation of Myers' bit-parallel Damerau-Levenshtein distance
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Configuration and data structures
#include "config.h"
#include "bitvector.h"
#include "hyrro_results.h"

// Utilities
#include "cpu_utils.h"
#include "gpu_utils.h"

// Core functionality
#include "hyrro_partition.h"
#include "hyrro_loading.h"
#include "hyrro_scheduler.h"
#include "hyrro_sender.h"
#include "levenshtein_kernel.cuh"

// ============================================================================
// GPU COMPUTATION THREAD
// ============================================================================

/**
 * @brief GPU computation thread function
 * 
 * This function runs in a separate thread to process the GPU portion of
 * the reference while FPGA processes its portion concurrently.
 * 
 * @param args Pointer to GPUArgs structure
 * @return NULL
 */
void* run_hyyro_gpu(void* args) {
    GPUArgs* gpu_args = (GPUArgs*)args;
    int num_queries = gpu_args->num_queries;
    int num_orig_refs = gpu_args->num_orig_refs;
    char** query_seqs = gpu_args->query_seqs;
    char** orig_refs = gpu_args->orig_refs;

    // ========== Prepare reference metadata ==========
    int* orig_ref_lens = (int*)malloc(num_orig_refs * sizeof(int));
    for (int i = 0; i < num_orig_refs; ++i) {
        orig_ref_lens[i] = (int)strlen(orig_refs[i]);
    }

    // ========== Compute optimal partitioning strategy ==========
    int chunk_size, partition_threshold;
    compute_partitioning_strategy(
        num_queries, num_orig_refs, orig_ref_lens,
        &chunk_size, &partition_threshold);

    printf("[GPU] Starting computation (Dynamic Chunk: %d, Threshold: %d, Threads: %d)\n", 
           chunk_size, partition_threshold, THREADS_PER_BLOCK);

    // ========== Partition references into chunks ==========
    int qlen0 = (int)strlen(query_seqs[0]);
    int overlap = qlen0 - 1;  // Standard overlap for pattern matching

    // Use computed dynamic chunk size for fine-grained parallelism
    PartitionedRefs part_refs = partition_references(
        orig_refs, orig_ref_lens, num_orig_refs,
        overlap, chunk_size, partition_threshold
    );

    int num_chunks = part_refs.num_chunks;
    printf("[GPU] Partitioned into %d chunks\n", num_chunks);

    // ========== Prepare query metadata ==========
    int* q_lens = (int*)malloc(num_queries * sizeof(int));
    for (int q = 0; q < num_queries; ++q) {
        q_lens[q] = (int)strlen(query_seqs[q]);
    }

    // ========== Precompute Eq bit-vector tables ==========
    bv_t* h_Eq_queries = buildEqTables(query_seqs, q_lens, num_queries);
    if (!h_Eq_queries) {
        fprintf(stderr, "ERROR: Failed to build Eq tables\n");
        free(q_lens);
        return NULL;
    }

    // ========== Pack sequences efficiently (no padding waste) ==========
    char* h_queries = NULL;
    char* h_refs = NULL;
    int* h_ref_offsets = NULL;
    size_t total_query_bytes = 0;
    size_t total_ref_bytes = 0;
    
    packSequencesEfficient(query_seqs, num_queries, 
                          part_refs.chunk_seqs, part_refs.chunk_lens, num_chunks,
                          &h_queries, &h_refs, &h_ref_offsets, &total_query_bytes, &total_ref_bytes);
    
    if (!h_queries || !h_refs || !h_ref_offsets) {
        fprintf(stderr, "ERROR: Out of memory for host buffers\n");
        free(h_Eq_queries);
        free(q_lens);
        return NULL;
    }

    // ========== Allocate GPU memory with ACTUAL sizes ==========
    GpuBuffers gpu_buffers;
    allocateGpuMemory(&gpu_buffers, num_queries, num_chunks, num_orig_refs, total_ref_bytes);

    // ========== Transfer data to GPU ==========
    transferToGpuEfficient(&gpu_buffers, h_Eq_queries, h_queries, h_refs,
                          q_lens, part_refs.chunk_lens, h_ref_offsets,
                          &part_refs, orig_ref_lens,
                          num_queries, num_chunks, num_orig_refs, total_query_bytes, total_ref_bytes);

    // ========== Build chunk-to-original mapping ==========
    int* orig_chunk_counts = NULL;
    int** orig_chunk_lists = NULL;
    build_orig_to_chunk_mapping(&part_refs, num_orig_refs, 
                                 &orig_chunk_counts, &orig_chunk_lists);

    // ========== Launch Damerau-Levenshtein kernel (two-pass approach) ==========
    int total_pairs = num_queries * num_chunks;
    int blocks = total_pairs;  // One block per query-chunk pair
    // Cap blocks at MAX_BLOCKS to respect GPU hardware limits
    if (blocks > MAX_BLOCKS) {
        fprintf(stderr, "[GPU] WARNING: Capping blocks from %d to %d (GPU limit)\n", blocks, MAX_BLOCKS);
        blocks = MAX_BLOCKS;
    }
    int threads = THREADS_PER_BLOCK;
    size_t shared_bytes = THREADS_PER_BLOCK * sizeof(bv_t);

    printf("[GPU] Launching kernel: %d blocks × %d threads (total: %d pairs)\n",
           blocks, threads, total_pairs);

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    
    float total_ms = 0.0f;

    for (int it = 0; it < LOOP_ITERATIONS; ++it) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        
        // Pass 1: Compute distances and find minimum scores
        levenshteinKernelOptimized<<<blocks, threads, shared_bytes>>>(
            num_queries, num_chunks, num_orig_refs,
            gpu_buffers.d_queries, gpu_buffers.d_qLens, gpu_buffers.d_EqQueries,
            gpu_buffers.d_refs, gpu_buffers.d_refLens, gpu_buffers.d_refOffsets,
            gpu_buffers.d_chunkStarts, gpu_buffers.d_chunkToOrig,
            gpu_buffers.d_origRefLens,
            gpu_buffers.d_pairDistances, gpu_buffers.d_pairZcounts,
            gpu_buffers.d_pairZindices,
            gpu_buffers.d_lowestScoreOrig, gpu_buffers.d_lowestCountOrig,
            gpu_buffers.d_lowestIndicesOrig, gpu_buffers.d_lastScoreOrig
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        
        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_start, ev_end));
        total_ms += iter_ms;
    }

    // Pass 2: Collect all positions with lowest score — run ONCE outside the
    // timing loop.  collectLowestIndicesKernel uses atomicAdd on
    // d_lowestCountOrig / d_lowestIndicesOrig, so calling it inside the loop
    // would accumulate counts 10× and overflow MAX_HITS slots.
    collectLowestIndicesKernel<<<blocks, threads, shared_bytes>>>(
        num_queries, num_chunks, num_orig_refs,
        gpu_buffers.d_queries, gpu_buffers.d_qLens, gpu_buffers.d_EqQueries,
        gpu_buffers.d_refs, gpu_buffers.d_refLens, gpu_buffers.d_refOffsets,
        gpu_buffers.d_chunkStarts, gpu_buffers.d_chunkToOrig,
        gpu_buffers.d_lowestScoreOrig,
        gpu_buffers.d_lowestCountOrig,
        gpu_buffers.d_lowestIndicesOrig
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    // ========== Copy results back from GPU ==========
    long long total_pair_chunks = (long long)num_queries * num_chunks;
    long long total_orig_pairs = (long long)num_queries * num_orig_refs;
    
    int* h_pair_distances = (int*)malloc((size_t)total_pair_chunks * sizeof(int));
    int* h_pair_zcounts = (int*)malloc((size_t)total_pair_chunks * sizeof(int));
    int* h_pair_zindices = (int*)malloc((size_t)total_pair_chunks * MAX_HITS * sizeof(int));
    int* h_lowest_score_orig = (int*)malloc((size_t)total_orig_pairs * sizeof(int));
    int* h_lowest_count_orig = (int*)malloc((size_t)total_orig_pairs * sizeof(int));
    int* h_lowest_indices_orig = (int*)malloc((size_t)total_orig_pairs * MAX_HITS * sizeof(int));
    int* h_last_score_orig = (int*)malloc((size_t)total_orig_pairs * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_pair_distances, gpu_buffers.d_pairDistances, 
                         (size_t)total_pair_chunks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zcounts, gpu_buffers.d_pairZcounts, 
                         (size_t)total_pair_chunks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zindices, gpu_buffers.d_pairZindices, 
                         (size_t)total_pair_chunks * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_score_orig, gpu_buffers.d_lowestScoreOrig, 
                         (size_t)total_orig_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_count_orig, gpu_buffers.d_lowestCountOrig, 
                         (size_t)total_orig_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_indices_orig, gpu_buffers.d_lowestIndicesOrig, 
                         (size_t)total_orig_pairs * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_last_score_orig, gpu_buffers.d_lastScoreOrig, 
                         (size_t)total_orig_pairs * sizeof(int), cudaMemcpyDeviceToHost));

    // ========== Last scores are correctly set by the GPU kernel ==========
    // h_last_score_orig already contains valid scores from d_lastScoreOrig:
    // - GPU-only path (small refs): kernel sets score at the final reference position
    // - Hetero path (large refs): FPGA final_score overrides this in the merge step
    // Do NOT reset to 0x7f7f7f7f here — that destroys the GPU-computed value.

    // ========== Allocate output arrays ==========
    gpu_args->output_lowest_scores = (int*)malloc(total_orig_pairs * sizeof(int));
    gpu_args->output_lowest_counts = (int*)malloc(total_orig_pairs * sizeof(int));
    gpu_args->output_last_scores = (int*)malloc(total_orig_pairs * sizeof(int));
    gpu_args->output_hit_counts = (int*)malloc(total_orig_pairs * sizeof(int));
    gpu_args->output_lowest_indices = (int**)malloc(total_orig_pairs * sizeof(int*));
    gpu_args->output_hit_indices = (int**)malloc(total_orig_pairs * sizeof(int*));

    memcpy(gpu_args->output_lowest_scores, h_lowest_score_orig, total_orig_pairs * sizeof(int));
    memcpy(gpu_args->output_lowest_counts, h_lowest_count_orig, total_orig_pairs * sizeof(int));
    memcpy(gpu_args->output_last_scores, h_last_score_orig, total_orig_pairs * sizeof(int));
    // ========== Aggregate results per original reference ==========
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long pair_idx = (long long)q * num_orig_refs + orig;

            // Aggregate hit indices (exact matches)
            DynamicIntArray hits;
            init_dynamic_array(&hits, 4096);

            for (int ci_idx = 0; ci_idx < orig_chunk_counts[orig]; ++ci_idx) {
                int chunk_id = orig_chunk_lists[orig][ci_idx];
                long long chunk_pair_idx = (long long)q * num_chunks + chunk_id;
                int zc = h_pair_zcounts[chunk_pair_idx];

                for (int k = 0; k < zc && k < MAX_HITS; ++k) {
                    int val = h_pair_zindices[chunk_pair_idx * MAX_HITS + k];
                    if (val >= 0) {
                        push_dynamic_array(&hits, val);
                    }
                }
            }

            sort_and_deduplicate(hits.data, &hits.size);
            gpu_args->output_hit_counts[pair_idx] = hits.size;
            
            if (hits.size > 0) {
                gpu_args->output_hit_indices[pair_idx] = (int*)malloc(hits.size * sizeof(int));
                memcpy(gpu_args->output_hit_indices[pair_idx], hits.data, hits.size * sizeof(int));
            } else {
                gpu_args->output_hit_indices[pair_idx] = NULL;
            }
            
            free_dynamic_array(&hits);

            // Process lowest score indices
            int lowest_cnt = h_lowest_count_orig[pair_idx];
            if (lowest_cnt > 0) {
                long long indices_base = pair_idx * MAX_HITS;
                int* lowest_arr = (int*)malloc(sizeof(int) * MIN(lowest_cnt, MAX_HITS));
                int valid_count = 0;

                for (int k = 0; k < MIN(lowest_cnt, MAX_HITS); ++k) {
                    int idx_val = h_lowest_indices_orig[indices_base + k];
                    if (idx_val >= 0) {
                        lowest_arr[valid_count++] = idx_val;
                    }
                }

                if (valid_count > 0) {
                    sort_and_deduplicate(lowest_arr, &valid_count);
                    gpu_args->output_lowest_indices[pair_idx] = (int*)malloc(valid_count * sizeof(int));
                    memcpy(gpu_args->output_lowest_indices[pair_idx], lowest_arr, valid_count * sizeof(int));
                    gpu_args->output_lowest_counts[pair_idx] = valid_count;
                } else {
                    gpu_args->output_lowest_indices[pair_idx] = NULL;
                    gpu_args->output_lowest_counts[pair_idx] = 0;
                }

                free(lowest_arr);
            } else {
                gpu_args->output_lowest_indices[pair_idx] = NULL;
            }
        }
    }

    gpu_args->avg_execution_time = (total_ms / LOOP_ITERATIONS) / 1000.0;  // Convert to seconds
    gpu_args->success = 1;

    // ========== Cleanup GPU memory ==========
    freeGpuMemory(&gpu_buffers);

    // ========== Cleanup host memory ==========
    free(h_Eq_queries);
    free(h_queries);
    free(h_refs);
    free(h_ref_offsets);
    free(h_pair_distances);
    free(h_pair_zcounts);
    free(h_pair_zindices);
    free(h_lowest_score_orig);
    free(h_lowest_count_orig);
    free(h_lowest_indices_orig);
    free(h_last_score_orig);
    free(q_lens);

    free_orig_to_chunk_mapping(orig_chunk_counts, orig_chunk_lists, num_orig_refs);
    free_partitioned_refs(&part_refs);
    free(orig_ref_lens);

    printf("[GPU] Computation complete\n");
    return NULL;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

/**
 * @brief Format an int array as a comma-separated string (no brackets).
 *        Returns a heap-allocated string; caller must free.
 */
static char* format_indices_csv(int* indices, int count) {
    if (count == 0) return strdup("N/A");
    // Each int is at most 11 chars + comma
    char* buf = (char*)malloc((size_t)count * 12 + 1);
    buf[0] = '\0';
    for (int i = 0; i < count; i++) {
        char tmp[20];
        sprintf(tmp, "%d", indices[i]);
        strcat(buf, tmp);
        if (i < count - 1) strcat(buf, ",");
    }
    return buf;
}

/**
 * @brief Create a directory if it doesn't already exist.
 */
static void ensure_dir(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0755);
    }
}

int main() {
    printf("\n=== Heterogeneous GPU-FPGA Damerau-Levenshtein ===\n");

    // ========== Scan input folders ==========
    int num_query_files = 0, num_ref_files = 0;
    char** query_files = get_files_from_folder(QUERY_FOLDER, &num_query_files);
    char** ref_files   = get_files_from_folder(REFERENCE_FOLDER, &num_ref_files);

    if (!query_files || num_query_files == 0) {
        fprintf(stderr, "No query files found in %s\n", QUERY_FOLDER);
        return EXIT_FAILURE;
    }
    if (!ref_files || num_ref_files == 0) {
        fprintf(stderr, "No reference files found in %s\n", REFERENCE_FOLDER);
        return EXIT_FAILURE;
    }

    printf("Found %d query file(s) and %d reference file(s) → %d pair(s)\n",
           num_query_files, num_ref_files, num_query_files * num_ref_files);

    // ========== Open CSV output ==========
    ensure_dir(RESULTS_FOLDER);
    FILE* csv = fopen(RESULTS_CSV_FILE, "w");
    if (!csv) {
        perror("Failed to create CSV file");
        return EXIT_FAILURE;
    }

    // Write header (matches results.csv format)
    fprintf(csv, "Hetero Run\n");
    fprintf(csv, "Query File,Query Length,Reference File,Reference Length,"
                 "Number of Hits,Hit Indexes,Lowest Score,Lowest Score Indexes,Last Score\n");

    // ========== Loop over every query file × reference file ==========
    for (int qf = 0; qf < num_query_files; qf++) {
        char query_path[1024];
        snprintf(query_path, sizeof(query_path), "%s/%s", QUERY_FOLDER, query_files[qf]);

        int num_queries = 0;
        char** query_seqs = load_queries_from_file(query_path, &num_queries);
        if (!query_seqs || num_queries == 0) continue;

        // Each file has exactly 1 query
        int query_len = (int)strlen(query_seqs[0]);

        // Save current query to FPGA split files so the FPGA sender
        // always uses the correct query for this iteration.
        save_queries_for_fpga(query_seqs, num_queries);

        for (int rf = 0; rf < num_ref_files; rf++) {
            char ref_path[1024];
            snprintf(ref_path, sizeof(ref_path), "%s/%s", REFERENCE_FOLDER, ref_files[rf]);

            int num_orig_refs = 0;
            char** orig_refs = load_references_from_file(ref_path, &num_orig_refs);
            if (!orig_refs || num_orig_refs == 0) continue;

            // Each file has exactly 1 reference
            printf("\n========================================================================================\n");
            printf("[Pair] %s  vs  %s\n", query_files[qf], ref_files[rf]);

            // ========== Allocate per-pair tracking arrays ==========
            GPUArgs *all_gpu_results = (GPUArgs*)calloc(num_orig_refs, sizeof(GPUArgs));
            FPGAResult **all_fpga_results = (FPGAResult**)calloc(num_orig_refs, sizeof(FPGAResult*));
            char **all_gpu_refs   = (char**)malloc(sizeof(char*) * num_orig_refs);
            int  *all_gpu_ref_lens = (int*)malloc(sizeof(int) * num_orig_refs);

            float gpu_ratio  = GPU_SPEED_RATIO;
            float fpga_ratio = FPGA_SPEED_RATIO;
            double total_gpu_time  = 0.0;
            double total_fpga_time = 0.0;

            // ========== Process each reference in the file ==========
            for (int ref_idx = 0; ref_idx < num_orig_refs; ref_idx++) {
                printf("\n[Processing Reference %d/%d] GPU: %.3f | FPGA: %.3f\n",
                       ref_idx + 1, num_orig_refs, gpu_ratio, fpga_ratio);

                int ref_len = (int)strlen(orig_refs[ref_idx]);
                split_reference_for_fpga_gpu(orig_refs[ref_idx], query_len,
                                             &all_gpu_refs[ref_idx],
                                             &all_gpu_ref_lens[ref_idx],
                                             ref_idx, gpu_ratio);

                double gpu_time = 0.0, fpga_time = 0.0;
                FPGAResult* fpga_results = NULL;

                if (ref_len >= MIN_REF_LENGTH_FOR_FPGA) {
                    GPUArgs gpu_args;
                    init_gpu_args(&gpu_args);
                    gpu_args.num_queries   = num_queries;
                    gpu_args.num_orig_refs = 1;
                    gpu_args.query_seqs    = query_seqs;
                    gpu_args.orig_refs     = &all_gpu_refs[ref_idx];

                    pthread_t gpu_thread;
                    if (pthread_create(&gpu_thread, NULL, run_hyyro_gpu, &gpu_args) != 0) {
                        fprintf(stderr, "ERROR: failed to create GPU thread\n");
                        return EXIT_FAILURE;
                    }

                    fpga_results = send_and_run_fpga_single_ref(num_queries, ref_idx, &fpga_time);

                    pthread_join(gpu_thread, NULL);
                    gpu_time = gpu_args.avg_execution_time;
                    all_gpu_results[ref_idx] = gpu_args;
                } else {
                    printf("[FPGA] Skipped (reference too small, GPU parallel chunks)\n");
                    GPUArgs gpu_args;
                    init_gpu_args(&gpu_args);
                    gpu_args.num_queries   = num_queries;
                    gpu_args.num_orig_refs = 1;
                    gpu_args.query_seqs    = query_seqs;
                    gpu_args.orig_refs     = &all_gpu_refs[ref_idx];

                    run_hyyro_gpu(&gpu_args);
                    gpu_time = gpu_args.avg_execution_time;
                    all_gpu_results[ref_idx] = gpu_args;
                }

                printf("[Timing] GPU: %.4fs | FPGA: %.4fs\n", gpu_time, fpga_time);
                total_gpu_time  += gpu_time;
                total_fpga_time += fpga_time;

                RatioThreadArgs ratio_args;
                pthread_t ratio_thread;
                start_ratio_thread(&ratio_thread, &ratio_args, gpu_time, fpga_time, gpu_ratio);
                gpu_ratio  = wait_for_ratio_thread(ratio_thread, &ratio_args);
                fpga_ratio = 1.0f - gpu_ratio;

                all_fpga_results[ref_idx] = fpga_results;
            }

            // ========== Merge results and write to CSV ==========
            printf("\n===== RESULTS for %s vs %s =====\n", query_files[qf], ref_files[rf]);

            int total_pairs = num_queries * num_orig_refs;
            int* merged_lowest_scores   = (int*)malloc(total_pairs * sizeof(int));
            int** merged_lowest_indices = (int**)malloc(total_pairs * sizeof(int*));
            int* merged_index_counts    = (int*)malloc(total_pairs * sizeof(int));

            for (int r = 0; r < num_orig_refs; r++) {
                for (int q = 0; q < num_queries; q++) {
                    long long idx     = (long long)q * num_orig_refs + r;
                    long long gpu_idx = q;

                    int orig_ref_len = (int)strlen(orig_refs[r]);

                    int lowest_gpu      = all_gpu_results[r].output_lowest_scores[gpu_idx];
                    int gpu_lowest_cnt  = all_gpu_results[r].output_lowest_counts[gpu_idx];
                    int lowest_fpga     = -1;
                    int fpga_lowest_cnt = 0;
                    int last_score      = -1;

                    if (all_fpga_results[r] != NULL) {
                        lowest_fpga     = all_fpga_results[r][q].lowest_score;
                        fpga_lowest_cnt = all_fpga_results[r][q].num_lowest_indexes;
                        last_score      = all_fpga_results[r][q].final_score;
                    } else {
                        last_score = all_gpu_results[r].output_last_scores[gpu_idx];
                    }

                    // Merge lowest scores
                    int merged_lowest = -1;
                    if (lowest_gpu != 0x7f7f7f7f && lowest_fpga != -1)
                        merged_lowest = MIN(lowest_gpu, lowest_fpga);
                    else if (lowest_gpu != 0x7f7f7f7f)
                        merged_lowest = lowest_gpu;
                    else
                        merged_lowest = lowest_fpga;

                    merged_lowest_scores[idx] = merged_lowest;

                    // Merge lowest score indices
                    int total_idx_cnt = 0;
                    if (merged_lowest != -1) {
                        if (lowest_gpu == merged_lowest)  total_idx_cnt += gpu_lowest_cnt;
                        if (lowest_fpga == merged_lowest) total_idx_cnt += fpga_lowest_cnt;
                    }
                    merged_index_counts[idx] = total_idx_cnt;

                    if (total_idx_cnt > 0) {
                        merged_lowest_indices[idx] = (int*)malloc(total_idx_cnt * sizeof(int));
                        int wp = 0;

                        if (lowest_gpu == merged_lowest &&
                            all_gpu_results[r].output_lowest_indices[gpu_idx] != NULL) {
                            for (int i = 0; i < gpu_lowest_cnt; i++)
                                merged_lowest_indices[idx][wp++] =
                                    all_gpu_results[r].output_lowest_indices[gpu_idx][i];
                        }
                        if (lowest_fpga == merged_lowest &&
                            all_fpga_results[r] != NULL &&
                            all_fpga_results[r][q].lowest_indexes != NULL) {
                            int fpga_offset = all_gpu_ref_lens[r] - query_len + 1;
                            for (int i = 0; i < fpga_lowest_cnt; i++)
                                merged_lowest_indices[idx][wp++] =
                                    (int)(all_fpga_results[r][q].lowest_indexes[i]) + fpga_offset;
                        }

                        qsort(merged_lowest_indices[idx], merged_index_counts[idx],
                              sizeof(int), compare_ints);

                        // Deduplicate
                        int unique = 1;
                        for (int i = 1; i < merged_index_counts[idx]; i++) {
                            if (merged_lowest_indices[idx][i] !=
                                merged_lowest_indices[idx][unique - 1])
                                merged_lowest_indices[idx][unique++] =
                                    merged_lowest_indices[idx][i];
                        }
                        merged_index_counts[idx] = unique;
                    } else {
                        merged_lowest_indices[idx] = NULL;
                    }

                    // Hits = positions where score == 0
                    int hit_count    = 0;
                    int* hit_indices = NULL;
                    if (merged_lowest == 0) {
                        hit_count    = merged_index_counts[idx];
                        hit_indices  = merged_lowest_indices[idx];
                    }

                    // ---- Console output ----
                    printf("\nPair: Q%d(%d) Vs R%d(%d)\n", q+1, query_len, r+1, orig_ref_len);
                    printf("Number of Hits: %d\n", hit_count);
                    if (hit_count == 0) printf("Hit Indexes: N/A\n");
                    else {
                        printf("Hit Indexes: [");
                        for (int i = 0; i < hit_count && i < 20; i++) {
                            if (i) printf(", ");
                            printf("%d", hit_indices[i]);
                        }
                        if (hit_count > 20) printf(", ... %d total", hit_count);
                        printf("]\n");
                    }
                    printf("Lowest Score: %d\n", merged_lowest);
                    printf("Lowest Score Indexes: ");
                    if (merged_index_counts[idx] == 0) printf("N/A\n");
                    else {
                        printf("[");
                        for (int i = 0; i < merged_index_counts[idx] && i < 20; i++) {
                            if (i) printf(", ");
                            printf("%d", merged_lowest_indices[idx][i]);
                        }
                        if (merged_index_counts[idx] > 20)
                            printf(", ... %d total", merged_index_counts[idx]);
                        printf("]\n");
                    }
                    printf("Last Score: %d\n", last_score);

                    // ---- CSV row ----
                    // Lowest Score column: "N/A" when there are hits (score==0)
                    char lowest_score_str[20];
                    if (hit_count > 0) strcpy(lowest_score_str, "N/A");
                    else               sprintf(lowest_score_str, "%d", merged_lowest);

                    char* hit_idx_str     = format_indices_csv(hit_indices, hit_count);
                    char* low_idx_str     = format_indices_csv(
                                               (hit_count > 0) ? NULL : merged_lowest_indices[idx],
                                               (hit_count > 0) ? 0    : merged_index_counts[idx]);

                    fprintf(csv, "%s,%d,%s,%d,%d,\"%s\",%s,\"%s\",%d\n",
                            query_files[qf], query_len,
                            ref_files[rf],   orig_ref_len,
                            hit_count,       hit_idx_str,
                            lowest_score_str, low_idx_str,
                            last_score);

                    free(hit_idx_str);
                    free(low_idx_str);
                }
            }

            printf("\n[Summary] GPU total: %.4fs | FPGA total: %.4fs\n",
                   total_gpu_time, total_fpga_time);

            // ========== Cleanup per-pair ==========
            for (int idx = 0; idx < total_pairs; idx++) {
                if (merged_lowest_indices[idx]) free(merged_lowest_indices[idx]);
            }
            free(merged_lowest_scores);
            free(merged_lowest_indices);
            free(merged_index_counts);

            for (int r = 0; r < num_orig_refs; r++) {
                if (all_gpu_refs[r]) free(all_gpu_refs[r]);
                free_gpu_results(&all_gpu_results[r], num_queries);
                free_fpga_results_single_ref(all_fpga_results[r], num_queries);
            }
            free(all_gpu_refs);
            free(all_gpu_ref_lens);
            free(all_gpu_results);
            free(all_fpga_results);

            for (int i = 0; i < num_orig_refs; i++) free(orig_refs[i]);
            free(orig_refs);
        } // end ref loop

        for (int i = 0; i < num_queries; i++) free(query_seqs[i]);
        free(query_seqs);
    } // end query loop

    fclose(csv);
    printf("\nResults saved to %s\n", RESULTS_CSV_FILE);

    free_file_list(query_files, num_query_files);
    free_file_list(ref_files,   num_ref_files);

    return EXIT_SUCCESS;
}
