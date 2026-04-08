/**
 * @file hyrro_results.h
 * @brief Data structures for storing computation results
 * 
 * This file defines structures for storing results from GPU and FPGA
 * computations, as well as functions for managing and merging results.
 */

#ifndef HYRRO_RESULTS_H
#define HYRRO_RESULTS_H

#include <stdlib.h>

// ============================================================================
// FPGA RESULT STRUCTURE
// ============================================================================

/**
 * @brief Results from a single FPGA computation
 * 
 * Contains timing information and distance metrics computed by the FPGA.
 */
typedef struct {
    double hw_exec_time_ms;          ///< Hardware execution time in milliseconds
    int final_score;                 ///< Edit distance at end of reference
    int lowest_score;                ///< Minimum edit distance found
    unsigned long *lowest_indexes;   ///< Positions where lowest score occurs
    int num_lowest_indexes;          ///< Number of positions with lowest score
    double overlay_load_ms;          ///< Time to load FPGA overlay (bitstream)
    double total_exec_ms;            ///< Total execution time including overhead
} FPGAResult;

/**
 * @brief Free memory allocated for FPGA result
 * @param res Pointer to FPGAResult structure
 */
static inline void free_fpga_result(FPGAResult *res) {
    if (res->lowest_indexes) {
        free(res->lowest_indexes);
        res->lowest_indexes = NULL;
    }
    res->num_lowest_indexes = 0;
}

/**
 * @brief Free array of FPGA results for a single reference
 * @param results Array of FPGAResult structures
 * @param num_queries Number of queries (size of array)
 */
static inline void free_fpga_results_single_ref(FPGAResult* results, int num_queries) {
    if (!results) return;
    
    for (int q = 0; q < num_queries; q++) {
        if (results[q].lowest_indexes) {
            free(results[q].lowest_indexes);
        }
    }
    free(results);
}

// ============================================================================
// GPU RESULT STRUCTURE
// ============================================================================

/**
 * @brief Arguments and results for GPU computation
 * 
 * This structure is used both as input (queries, references) and output
 * (distances, hit positions) for GPU processing threads.
 */
typedef struct {
    // ========== INPUT PARAMETERS ==========
    int num_queries;                 ///< Number of query sequences
    int num_orig_refs;               ///< Number of reference sequences
    char **query_seqs;               ///< Array of query sequence strings
    char **orig_refs;                ///< Array of reference sequence strings
    
    // ========== OUTPUT PARAMETERS ==========
    int* output_lowest_scores;       ///< Array: num_queries * num_orig_refs
    int** output_lowest_indices;     ///< Array of arrays for each query-ref pair
    int* output_lowest_counts;       ///< Array: num_queries * num_orig_refs
    int* output_last_scores;         ///< Array: num_queries * num_orig_refs
    int** output_hit_indices;        ///< Array of arrays for exact matches
    int* output_hit_counts;          ///< Array: num_queries * num_orig_refs
    
    // ========== PERFORMANCE METRICS ==========
    double avg_execution_time;       ///< Average execution time in seconds
    int success;                     ///< 1 = success, 0 = failure
} GPUArgs;

/**
 * @brief Initialize GPU arguments structure
 * @param args Pointer to GPUArgs structure to initialize
 * 
 * Sets all pointers to NULL and counters to 0.
 */
static inline void init_gpu_args(GPUArgs* args) {
    args->num_queries = 0;
    args->num_orig_refs = 0;
    args->query_seqs = NULL;
    args->orig_refs = NULL;
    args->output_lowest_scores = NULL;
    args->output_lowest_indices = NULL;
    args->output_lowest_counts = NULL;
    args->output_last_scores = NULL;
    args->output_hit_indices = NULL;
    args->output_hit_counts = NULL;
    args->avg_execution_time = 0.0;
    args->success = 0;
}

/**
 * @brief Free memory allocated for GPU results
 * @param args Pointer to GPUArgs structure
 * @param num_pairs Total number of query-reference pairs
 * 
 * Frees all dynamically allocated arrays in the GPU results.
 */
static inline void free_gpu_results(GPUArgs* args, int num_pairs) {
    if (!args) return;
    
    // Free simple arrays
    if (args->output_lowest_scores) free(args->output_lowest_scores);
    if (args->output_lowest_counts) free(args->output_lowest_counts);
    if (args->output_last_scores) free(args->output_last_scores);
    if (args->output_hit_counts) free(args->output_hit_counts);
    
    // Free arrays of arrays
    if (args->output_lowest_indices) {
        for (int i = 0; i < num_pairs; i++) {
            if (args->output_lowest_indices[i]) {
                free(args->output_lowest_indices[i]);
            }
        }
        free(args->output_lowest_indices);
    }
    
    if (args->output_hit_indices) {
        for (int i = 0; i < num_pairs; i++) {
            if (args->output_hit_indices[i]) {
                free(args->output_hit_indices[i]);
            }
        }
        free(args->output_hit_indices);
    }
    
    // Reset structure
    init_gpu_args(args);
}

// ============================================================================
// MERGED RESULT STRUCTURE
// ============================================================================

/**
 * @brief Merged results from both GPU and FPGA computations
 * 
 * Used to combine results from heterogeneous processing.
 */
typedef struct {
    int lowest_score;                ///< Best (minimum) edit distance found
    int* lowest_indices;             ///< Positions where lowest score occurs
    int num_lowest_indices;          ///< Number of positions with lowest score
    int num_exact_matches;           ///< Number of exact matches (score = 0)
    int last_score;                  ///< Edit distance at end of reference
} MergedResult;

/**
 * @brief Initialize merged result structure
 * @param result Pointer to MergedResult structure to initialize
 */
static inline void init_merged_result(MergedResult* result) {
    result->lowest_score = -1;
    result->lowest_indices = NULL;
    result->num_lowest_indices = 0;
    result->num_exact_matches = 0;
    result->last_score = -1;
}

/**
 * @brief Free memory allocated for merged result
 * @param result Pointer to MergedResult structure
 */
static inline void free_merged_result(MergedResult* result) {
    if (result->lowest_indices) {
        free(result->lowest_indices);
        result->lowest_indices = NULL;
    }
    result->num_lowest_indices = 0;
}

#endif // HYRRO_RESULTS_H