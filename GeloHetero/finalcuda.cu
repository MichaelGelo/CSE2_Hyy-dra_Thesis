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

    printf("[GPU] Starting computation (Chunk: %d, Threshold: %d, Threads: %d)\n", 
           CHUNK_SIZE, PARTITION_THRESHOLD, THREADS_PER_BLOCK);

    // ========== Prepare reference metadata ==========
    int* orig_ref_lens = (int*)malloc(num_orig_refs * sizeof(int));
    for (int i = 0; i < num_orig_refs; ++i) {
        orig_ref_lens[i] = (int)strlen(orig_refs[i]);
    }

    // ========== Partition references into chunks ==========
    int qlen0 = (int)strlen(query_seqs[0]);
    int overlap = qlen0 - 1;  // Standard overlap for pattern matching

    PartitionedRefs part_refs = partition_references(
        orig_refs, orig_ref_lens, num_orig_refs,
        overlap, CHUNK_SIZE, PARTITION_THRESHOLD
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
    size_t total_ref_bytes = 0;
    
    packSequencesEfficient(query_seqs, num_queries, 
                          part_refs.chunk_seqs, part_refs.chunk_lens, num_chunks,
                          &h_queries, &h_refs, &h_ref_offsets, &total_ref_bytes);
    
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
                          num_queries, num_chunks, num_orig_refs, total_ref_bytes);

    // ========== Build chunk-to-original mapping ==========
    int* orig_chunk_counts = NULL;
    int** orig_chunk_lists = NULL;
    build_orig_to_chunk_mapping(&part_refs, num_orig_refs, 
                                 &orig_chunk_counts, &orig_chunk_lists);

    // ========== Launch Damerau-Levenshtein kernel (two-pass approach) ==========
    int total_pairs = num_queries * num_chunks;
    int blocks = total_pairs;  // One block per query-chunk pair
    int threads = THREADS_PER_BLOCK;
    size_t shared_bytes = 256 * sizeof(bv_t);

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
        
        // Pass 2: Collect all positions with lowest score
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
        
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        
        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_start, ev_end));
        total_ms += iter_ms;
    }

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

    // ========== Correct last scores on host ==========
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long orig_pair_idx = (long long)q * num_orig_refs + orig;
            int orig_len = orig_ref_lens[orig];
            // int correct_last_score = 0x7f7f7f7f;  // Unused - will be set from FPGA

            // Find chunk that ends at original reference length
            // (Last score correction - currently handled by FPGA final_score)
            for (int ci_idx = 0; ci_idx < orig_chunk_counts[orig]; ++ci_idx) {
                int chunk_id = orig_chunk_lists[orig][ci_idx];
                int chunk_start = part_refs.chunk_starts[chunk_id];
                int chunk_len = part_refs.chunk_lens[chunk_id];

                if (chunk_start + chunk_len == orig_len) {
                    // This chunk ends at the original reference length
                    // (Last score will be taken from FPGA final_score)
                    break;
                }
            }

            h_last_score_orig[orig_pair_idx] = 0x7f7f7f7f;
        }
    }

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

int main() {
    printf("\n=== Heterogeneous GPU-FPGA Damerau-Levenshtein ===\n");
    printf("Loading data...\n");

    // ========== Load data ==========
    int num_queries = 0, num_orig_refs = 0;
    char **query_seqs = load_queries(&num_queries);
    if (!query_seqs || num_queries == 0) {
        fprintf(stderr, "Failed to load queries\n");
        return EXIT_FAILURE;
    }

    char **orig_refs = load_references(&num_orig_refs);
    if (!orig_refs || num_orig_refs <= 0) {
        fprintf(stderr, "Failed to load references\n");
        return EXIT_FAILURE;
    }

    printf("Loaded: %d queries, %d references\n", num_queries, num_orig_refs);

    int query_len = (int)strlen(query_seqs[0]);

    // ========== Allocate tracking arrays ==========
    GPUArgs *all_gpu_results = (GPUArgs*)calloc(num_orig_refs, sizeof(GPUArgs));
    FPGAResult **all_fpga_results = (FPGAResult**)calloc(num_orig_refs, sizeof(FPGAResult*));
    
    char **all_gpu_refs = (char**)malloc(sizeof(char*) * num_orig_refs);
    int *all_gpu_ref_lens = (int*)malloc(sizeof(int) * num_orig_refs);

    float gpu_ratio = GPU_SPEED_RATIO;
    float fpga_ratio = FPGA_SPEED_RATIO;

    double total_gpu_time = 0.0;
    double total_fpga_time = 0.0;

    // ========== Process each reference ==========
    for (int ref_idx = 0; ref_idx < num_orig_refs; ref_idx++) {
        printf("\n========================================================================================");
        printf("\n[Processing Reference %d/%d] GPU: %.3f | FPGA: %.3f\n", 
               ref_idx + 1, num_orig_refs, gpu_ratio, fpga_ratio);

        // Split reference
        split_reference_for_fpga_gpu(orig_refs[ref_idx], query_len,
                                    &all_gpu_refs[ref_idx],
                                    &all_gpu_ref_lens[ref_idx],
                                    ref_idx,
                                    gpu_ratio);

        // Start GPU thread
        GPUArgs gpu_args;
        init_gpu_args(&gpu_args);
        gpu_args.num_queries = num_queries;
        gpu_args.num_orig_refs = 1;
        gpu_args.query_seqs = query_seqs;
        gpu_args.orig_refs = &all_gpu_refs[ref_idx];

        pthread_t gpu_thread;
        if (pthread_create(&gpu_thread, NULL, run_hyyro_gpu, &gpu_args) != 0) {
            fprintf(stderr, "ERROR: failed to create GPU thread\n");
            return EXIT_FAILURE;
        }

        // Run FPGA (blocking)
        double fpga_time = 0.0;
        FPGAResult* fpga_results = send_and_run_fpga_single_ref(num_queries, ref_idx, &fpga_time);

        // Wait for GPU
        pthread_join(gpu_thread, NULL);
        double gpu_time = gpu_args.avg_execution_time;

        printf("[Timing] GPU: %.4fs | FPGA: %.4fs\n", gpu_time, fpga_time);

        total_gpu_time += gpu_time;
        total_fpga_time += fpga_time;

        // Update ratio for next iteration
        RatioThreadArgs ratio_args;
        pthread_t ratio_thread;
        
        start_ratio_thread(&ratio_thread, &ratio_args, gpu_time, fpga_time, gpu_ratio);
        gpu_ratio = wait_for_ratio_thread(ratio_thread, &ratio_args);
        fpga_ratio = 1.0f - gpu_ratio;

        // Store results
        all_gpu_results[ref_idx] = gpu_args;
        all_fpga_results[ref_idx] = fpga_results;
    }

    // ========== Print final results ==========
    // ========== Print final results ==========
    printf("\n===== RESULTS =====\n");

    int total_pairs = num_queries * num_orig_refs;
    int* merged_lowest_scores = (int*)malloc(total_pairs * sizeof(int));
    int** merged_lowest_indices = (int**)malloc(total_pairs * sizeof(int*));
    int* merged_index_counts = (int*)malloc(total_pairs * sizeof(int));

    for (int r = 0; r < num_orig_refs; r++){
        for (int q = 0; q < num_queries; q++){
            long long idx = (long long)q * num_orig_refs + r;

            printf("\n--- Query %d vs Reference %d ---\n", q+1, r+1);

            long long gpu_idx = q;  // GPU processes one ref at a time

            int lowest_gpu = all_gpu_results[r].output_lowest_scores[gpu_idx];
            int gpu_lowest_count = all_gpu_results[r].output_lowest_counts[gpu_idx];

            int lowest_fpga = all_fpga_results[r][q].lowest_score;
            int fpga_lowest_count = all_fpga_results[r][q].num_lowest_indexes;

            // CRITICAL FIX: Use FPGA's final_score as the true last score
            // because FPGA processes to the actual end of the reference
            int last_score = all_fpga_results[r][q].final_score;

            // printf("GPU: %d (%d), FPGA: %d (%d)\n", lowest_gpu, gpu_lowest_count, lowest_fpga, fpga_lowest_count);

            // ========== Merge lowest scores ==========
            int final_merged_lowest_score = -1;
            if (lowest_gpu != 0x7f7f7f7f && lowest_fpga != -1) {
                final_merged_lowest_score = MIN(lowest_gpu, lowest_fpga);
            } else if (lowest_gpu != 0x7f7f7f7f) {
                final_merged_lowest_score = lowest_gpu;
            } else {
                final_merged_lowest_score = lowest_fpga;
            }

            merged_lowest_scores[idx] = final_merged_lowest_score;

            // ========== Merge lowest score indices ==========
            int total_index_count = 0;
            if (final_merged_lowest_score != -1) {
                if (lowest_gpu == final_merged_lowest_score) {
                    total_index_count += gpu_lowest_count;
                }
                if (lowest_fpga == final_merged_lowest_score) {
                    total_index_count += fpga_lowest_count;
                }
            }

            merged_index_counts[idx] = total_index_count;

            if (total_index_count > 0) {
                merged_lowest_indices[idx] = (int*)malloc(total_index_count * sizeof(int));
                int write_pos = 0;

                // Add GPU indices if they match the merged lowest score
                if (lowest_gpu == final_merged_lowest_score &&
                    all_gpu_results[r].output_lowest_indices[gpu_idx] != NULL) {
                    for (int i = 0; i < gpu_lowest_count; i++) {
                        merged_lowest_indices[idx][write_pos++] =
                            all_gpu_results[r].output_lowest_indices[gpu_idx][i];
                    }
                }

                // Add FPGA indices with proper offset if they match
                if (lowest_fpga == final_merged_lowest_score &&
                    all_fpga_results[r][q].lowest_indexes != NULL) {
                    int gpu_len = all_gpu_ref_lens[r];
                    int fpga_offset = gpu_len - query_len + 1;
                    for (int i = 0; i < fpga_lowest_count; i++) {
                        merged_lowest_indices[idx][write_pos++] =
                            (int)(all_fpga_results[r][q].lowest_indexes[i]) + fpga_offset;
                    }
                }

                // Sort and deduplicate
                qsort(merged_lowest_indices[idx], merged_index_counts[idx],
                      sizeof(int), compare_ints);
                
                // Remove duplicates in-place
                int unique_count = 1;
                for (int i = 1; i < merged_index_counts[idx]; i++) {
                    if (merged_lowest_indices[idx][i] != 
                        merged_lowest_indices[idx][unique_count - 1]) {
                        merged_lowest_indices[idx][unique_count++] = 
                            merged_lowest_indices[idx][i];
                    }
                }
                merged_index_counts[idx] = unique_count;
                
            } else {
                merged_lowest_indices[idx] = NULL;
            }

            // Count exact matches (score = 0)
            int hit_count = 0;
            if (merged_lowest_scores[idx] == 0) {
                hit_count = merged_index_counts[idx];
            }

            printf("Lowest: %d | Hits: %d | Positions: %d | Last: %d\n", 
                   merged_lowest_scores[idx], hit_count, 
                   merged_index_counts[idx], last_score);

            if (merged_index_counts[idx] > 0 && merged_index_counts[idx] <= 20) {
                printf("Positions: [");
                for (int i = 0; i < merged_index_counts[idx]; i++) {
                    if (i > 0) printf(", ");
                    printf("%d", merged_lowest_indices[idx][i]);
                }
                printf("]\n");
            } else if (merged_index_counts[idx] > 20) {
                printf("Positions: [%d, %d, ... %d positions total]\n",
                       merged_lowest_indices[idx][0], 
                       merged_lowest_indices[idx][1],
                       merged_index_counts[idx]);
            }
        }
    }

    printf("\n===== SUMMARY =====\n");
    printf("Total GPU: %.4fs | Total FPGA: %.4fs\n", total_gpu_time, total_fpga_time);
    printf("Final ratio - GPU: %.3f | FPGA: %.3f\n", gpu_ratio, fpga_ratio);
    printf("===================\n");

    // ========== Cleanup ==========
    for (int idx = 0; idx < total_pairs; idx++) {
        if (merged_lowest_indices[idx]) {
            free(merged_lowest_indices[idx]);
        }
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

    for (int i = 0; i < num_queries; i++) {
        free(query_seqs[i]);
    }
    free(query_seqs);

    for (int i = 0; i < num_orig_refs; i++) {
        free(orig_refs[i]);
    }
    free(orig_refs);

    return EXIT_SUCCESS;
}