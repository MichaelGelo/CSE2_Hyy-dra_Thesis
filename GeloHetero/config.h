// ============================================================================
// CONFIG.H - CONFIGURATION PARAMETERS
// Central location for all user-configurable parameters
// ============================================================================

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// FILE PATHS
// Input files for queries and references
// ============================================================================
#define QUERY_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/que1_192.fasta"
#define REFERENCE_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/Human.fasta"
#define FPGA_OUTPUT_DIR "./fpga_splits/"

// ============================================================================
// FPGA CONNECTION SETTINGS
// SSH and network configuration for FPGA communication
// ============================================================================
#define USERNAME        "xilinx"
#define FPGA_IP         "192.168.2.99"
#define FPGA_PORT       5000                // TCP port for direct socket communication
#define REMOTE_PATH     "/home/xilinx/jupyter_notebooks/updatedbit"
#define FPGA_SCRIPT     "fpga_code.py"
#define SSH_KEY         "~/.ssh/id_rsa_fpga"

#define HOST_FPGA_REF_DIR   "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/GeloHetero/fpga_splits"
#define HOST_FPGA_QUERY_DIR "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/GeloHetero/fpga_splits"


// ============================================================================
// UTILITY MACROS
// Common macros used throughout the codebase
// ============================================================================
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// ============================================================================
// GPU KERNEL PARAMETERS
// Threading and memory configuration for CUDA kernels
// ============================================================================
#define THREADS_PER_BLOCK 256           // Number of threads per GPU block
#define LOOP_ITERATIONS 10              // Number of kernel iterations for timing
#define MAX_LENGTH (1 << 24)            // Per-slot buffer size (16 MB)
#define MAX_QUERY_LENGTH (1 << 14)      // Max query length (16KB)
#define MAX_HITS 1024                   // Maximum number of hit indices to store

// ============================================================================
// PARTITIONING PARAMETERS
// Controls how large references are split into chunks
// ============================================================================
#define CHUNK_SIZE 166700               // Default chunk size for large references
#define PARTITION_THRESHOLD 1000000     // References > this size get partitioned

// JETSON ORIN GPU OPTIMIZATION TARGETS
#define SM_COUNT 64                     // Number of Streaming Multiprocessors
#define BLOCKS_PER_SM 12                // Blocks per SM for optimal occupancy
#define TARGET_BLOCKS (SM_COUNT * BLOCKS_PER_SM)  // ~768 blocks for maximum parallelism
#define MAX_BLOCKS 512                  // Safety cap: conservative limit to prevent GPU overflow
#define MIN_CHUNK_SIZE 5000             // Minimum chunk size (enables fine-grained parallelism)
#define MAX_CHUNK_SIZE 300000           // Maximum chunk size (memory constraint)

// ============================================================================
// ADAPTIVE SCHEDULING PARAMETERS
// Controls GPU/FPGA workload distribution
// ============================================================================
#define GPU_SPEED_RATIO  0.7f         // Initial GPU workload ratio (0.0-1.0)
#define FPGA_SPEED_RATIO 0.3f          // Initial FPGA workload ratio (0.0-1.0)

#define MIN_GPU_RATIO 0.05f             // Minimum GPU ratio (always do some GPU work)
#define MAX_GPU_RATIO 0.95f             // Maximum GPU ratio (always do some FPGA work)
#define RATIO_SMOOTHING_ALPHA 0.5f      // Adaptation speed (0=no change, 1=immediate)

#define MIN_REF_LENGTH_FOR_FPGA 1000000 // References below this length run on GPU only (parallel chunks)

// ============================================================================
// BIT-VECTOR CONFIGURATION
// Number of 64-bit words in bit vector (4 words = 256 bits)
// ============================================================================
#define BV_WORDS 4                      // Supports query lengths up to 256 chars

// ============================================================================
// FILE I/O LIMITS
// Maximum sizes for file reading and parsing
// ============================================================================
#define MAX_FILE_LENGTH (1 << 29)       // 536 MB max file size (supports up to 500M files)
#define MAX_LINE_LENGTH (1 << 14)       // 16 KB max line length in FASTA

// ============================================================================
// DYNAMIC PARTITIONING STRATEGY (JETSON ORIN TUNED)
// ============================================================================
// Computes optimal chunk size and partition threshold for GPU reference partitioning.
// This enables fine-grained parallelism for small references by creating many chunks.
// Example: 845K reference → 170 chunks @ 5KB each → 170 parallel GPU blocks
//
static inline void compute_partitioning_strategy(
    int num_queries, int num_refs, const int* ref_lengths,
    int* out_chunk_size, int* out_partition_threshold)
{
    if (num_queries <= 0 || num_refs <= 0) {
        // Invalid input, use defaults
        *out_chunk_size = CHUNK_SIZE;
        *out_partition_threshold = PARTITION_THRESHOLD;
        return;
    }

    // Calculate total and average reference length
    long long total_ref_length = 0;
    for (int i = 0; i < num_refs; ++i) {
        total_ref_length += ref_lengths[i];
    }
    int avg_ref_length = (int)(total_ref_length / num_refs);

    // Determine the target number of chunks per query to maximize GPU utilization
    int target_chunks = (TARGET_BLOCKS + num_queries - 1) / num_queries;
    if (target_chunks < 1) target_chunks = 1;
    // Cap to MAX_BLOCKS to prevent GPU overflow
    if (target_chunks > MAX_BLOCKS) target_chunks = MAX_BLOCKS;

    // Compute chunk size based on average reference length and target parallelism
    int chunk_size = avg_ref_length / target_chunks;

    // Clamp chunk size to allowed range
    if (chunk_size < MIN_CHUNK_SIZE) chunk_size = MIN_CHUNK_SIZE;
    if (chunk_size > MAX_CHUNK_SIZE) chunk_size = MAX_CHUNK_SIZE;

    // Set output values
    *out_chunk_size = chunk_size;
    // For GPU-only mode, always partition (threshold = chunk_size)
    *out_partition_threshold = chunk_size;
}

#endif // CONFIG_H