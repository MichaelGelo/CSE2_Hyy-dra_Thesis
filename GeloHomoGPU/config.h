// ============================================================================
// config.h
// Fast, GPU-biased configuration for Hyyrö bit-vector algorithm
// ============================================================================

#ifndef HYRRO_CONFIG_H
#define HYRRO_CONFIG_H

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

// ============================================================================
// FILE PATHS
// ============================================================================
#define QUERY_FILE     "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/que1_256.fasta"
#define REFERENCE_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/ref_1M.fasta"

// ============================================================================
// ALGORITHM LIMITS
// ============================================================================
#define MAX_QUERY_LENGTH 512
#define MAX_HITS 1024

// ============================================================================
// GPU EXECUTION PARAMETERS
// ============================================================================
#define THREADS_PER_BLOCK 256
#define LOOP_ITERATIONS 10

// ============================================================================
// PARTITIONING TARGETS (JETSON ORIN SPECIFIC)
// ============================================================================
#define SM_COUNT 64
#define BLOCKS_PER_SM 12
#define TARGET_BLOCKS (SM_COUNT * BLOCKS_PER_SM)  // ~768 blocks

#define MIN_CHUNK_SIZE 5000
#define MAX_CHUNK_SIZE 300000

// ============================================================================
// UTILITY MACROS
// ============================================================================
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// ============================================================================
// GPU MEMORY ESTIMATION (UPPER BOUND, LOGGING ONLY)
// ============================================================================
static inline size_t estimate_gpu_memory_needed(
    int num_queries, int num_chunks, int num_orig_refs, int avg_chunk_len)
{
    size_t mem = 0;
    mem += (size_t)num_queries * 256 * sizeof(uint64_t) * 4;  // Eq tables
    mem += (size_t)num_queries * MAX_QUERY_LENGTH;             // Queries
    mem += (size_t)num_chunks * avg_chunk_len;                 // Reference chunks
    mem += (size_t)num_queries * num_chunks * 10 * sizeof(int);      // Metadata
    mem += (size_t)num_queries * num_orig_refs * 10 * sizeof(int);   // Metadata
    return mem;
}

// ============================================================================
// PARTITIONING STRATEGY (JETSON ORIN TUNED)
// ============================================================================
//
// Computes optimal chunk size and partition threshold for GPU reference partitioning.
// - num_queries: Number of query sequences
// - num_refs: Number of reference sequences
// - ref_lengths: Array of reference sequence lengths
// - out_chunk_size: Output pointer for computed chunk size
// - out_partition_threshold: Output pointer for partition threshold
//
static inline void compute_partitioning_strategy(
    int num_queries, int num_refs, const int* ref_lengths,
    int* out_chunk_size, int* out_partition_threshold)
{
    if (num_queries <= 0 || num_refs <= 0) {
        // Invalid input, do nothing
        return;
    }

    // Calculate total and average reference length
    long long total_ref_length = 0;
    for (int i = 0; i < num_refs; ++i) {
        total_ref_length += ref_lengths[i];
    }
    int avg_ref_length = (int)(total_ref_length / num_refs);

    // Determine the target number of chunks per query
    int target_chunks = (TARGET_BLOCKS + num_queries - 1) / num_queries;
    if (target_chunks < 1) target_chunks = 1;

    // Compute initial chunk size
    int chunk_size = avg_ref_length / target_chunks;

    // Clamp chunk size to allowed range
    if (chunk_size < MIN_CHUNK_SIZE) chunk_size = MIN_CHUNK_SIZE;
    if (chunk_size > MAX_CHUNK_SIZE) chunk_size = MAX_CHUNK_SIZE;

    // Set output values
    *out_chunk_size = chunk_size;
    // Partition threshold can be set to chunk_size, or tuned separately if needed
    *out_partition_threshold = chunk_size;

    // Estimate total number of chunks and memory usage for logging
    int estimated_chunks = 0;
    long long total_chunk_bytes = 0;
    for (int i = 0; i < num_refs; ++i) {
        int chunks = (ref_lengths[i] + chunk_size - 1) / chunk_size;
        estimated_chunks += chunks;
        total_chunk_bytes += (long long)chunks * (chunk_size + 256); // 256 for padding/overhead
    }

    int estimated_blocks = num_queries * estimated_chunks;
    int avg_chunk_len = estimated_chunks > 0 ? (int)(total_chunk_bytes / estimated_chunks) : 0;
    size_t estimated_mem = estimate_gpu_memory_needed(
        num_queries, estimated_chunks, num_refs, avg_chunk_len);

    // Logging for tuning and debugging
    printf("[Partitioning] chunk_size=%d, partition_threshold=%d\n", chunk_size, *out_partition_threshold);
    printf("[Partitioning] Estimated GPU blocks=%d (target ~%d)\n", estimated_blocks, TARGET_BLOCKS);
    printf("[Partitioning] Estimated GPU memory=%.2f MB\n", estimated_mem/(1024.0*1024.0));
}

#endif // HYRRO_CONFIG_H
