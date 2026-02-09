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
#define QUERY_FILE "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/que1_256.fasta"
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
#define CHUNK_SIZE 166700               // Size of each chunk in characters
#define PARTITION_THRESHOLD 1000000     // References > this size get partitioned

// ============================================================================
// ADAPTIVE SCHEDULING PARAMETERS
// Controls GPU/FPGA workload distribution
// ============================================================================
#define GPU_SPEED_RATIO  0.7f         // Initial GPU workload ratio (0.0-1.0)
#define FPGA_SPEED_RATIO 0.3f          // Initial FPGA workload ratio (0.0-1.0)

#define MIN_GPU_RATIO 0.05f             // Minimum GPU ratio (always do some GPU work)
#define MAX_GPU_RATIO 0.95f             // Maximum GPU ratio (always do some FPGA work)
#define RATIO_SMOOTHING_ALPHA 0.5f      // Adaptation speed (0=no change, 1=immediate)

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

#endif // CONFIG_H