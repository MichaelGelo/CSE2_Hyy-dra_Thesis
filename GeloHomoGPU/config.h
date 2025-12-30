// hyyro_config.h
// Configuration constants and file paths for Hyyrö algorithm

#ifndef HYRRO_CONFIG_H
#define HYRRO_CONFIG_H

// ============================================================================
// FILE PATHS
// ============================================================================
#define QUERY_FILE     "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/que1_256.fasta"
#define REFERENCE_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/ref_1M.fasta"

// ============================================================================
// MEMORY LIMITS
// ============================================================================
#define MAX_LENGTH (1 << 24)   // Per-slot buffer size (16MB, must be >= longest chunk)
#define MAX_HITS 1024          // Maximum number of hit positions to track per pair

// ============================================================================
// GPU PARAMETERS
// ============================================================================
#define THREADS_PER_BLOCK 256  // Number of threads per CUDA block
#define LOOP_ITERATIONS 10     // Number of kernel iterations for timing

// ============================================================================
// PARTITIONING PARAMETERS
// ============================================================================
#define CHUNK_SIZE 50000            // Size of each reference chunk
#define PARTITION_THRESHOLD 100000  // Minimum reference length to trigger partitioning

// ============================================================================
// UTILITY MACROS
// ============================================================================
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#endif // HYYRO_CONFIG_H