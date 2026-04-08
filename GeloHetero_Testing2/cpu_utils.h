/**
 * @file cpu_utils.h
 * @brief CPU-side utility functions for data processing
 * 
 * This file provides host-side utilities for timing, sorting, deduplication,
 * and result aggregation operations.
 */

#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// TIMING UTILITIES
// ============================================================================

/**
 * @brief Get current time in seconds with microsecond precision
 * @return Current time as double (seconds since epoch)
 * 
 * Uses gettimeofday for high-precision timing measurements.
 */
static inline double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/**
 * @brief Calculate elapsed time between two timeval structures
 * @param start Start time
 * @param end End time
 * @return Elapsed time in seconds
 */
static inline double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

// ============================================================================
// SORTING AND COMPARISON
// ============================================================================

/**
 * @brief Comparison function for integer sorting
 * @param a Pointer to first integer
 * @param b Pointer to second integer
 * @return Negative if a < b, 0 if a == b, positive if a > b
 * 
 * Used with qsort() to sort integer arrays in ascending order.
 */
static inline int compare_ints(const void *a, const void *b) {
    int av = *(const int*)a;
    int bv = *(const int*)b;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
}

/**
 * @brief Sort and deduplicate an array of integers in-place
 * @param arr Array to sort and deduplicate
 * @param count Pointer to array size (updated with new size)
 * 
 * Sorts the array in ascending order and removes duplicate values.
 * Updates *count to reflect the new size after deduplication.
 * 
 * Example:
 *   arr = [5, 2, 8, 2, 5, 1], count = 6
 *   After: arr = [1, 2, 5, 8], count = 4
 */
static inline void sort_and_deduplicate(int* arr, int* count) {
    if (*count <= 0) return;
    
    // Sort array
    qsort(arr, *count, sizeof(int), compare_ints);
    
    // Deduplicate: keep only first occurrence of each value
    int write_pos = 1;
    for (int i = 1; i < *count; i++) {
        if (arr[i] != arr[write_pos - 1]) {
            arr[write_pos++] = arr[i];
        }
    }
    
    *count = write_pos;
}

// ============================================================================
// DYNAMIC ARRAY UTILITIES
// ============================================================================

/**
 * @brief Dynamic integer array structure
 * 
 * Automatically grows as needed when elements are added.
 */
typedef struct {
    int* data;          ///< Array data
    int size;           ///< Current number of elements
    int capacity;       ///< Allocated capacity
} DynamicIntArray;

/**
 * @brief Initialize dynamic array
 * @param arr Pointer to DynamicIntArray structure
 * @param initial_capacity Initial capacity (0 for default)
 */
static inline void init_dynamic_array(DynamicIntArray* arr, int initial_capacity) {
    if (initial_capacity <= 0) initial_capacity = 4096;
    arr->data = (int*)malloc(initial_capacity * sizeof(int));
    arr->size = 0;
    arr->capacity = initial_capacity;
}

/**
 * @brief Add element to dynamic array
 * @param arr Pointer to DynamicIntArray structure
 * @param value Value to add
 * 
 * Automatically doubles capacity if needed.
 */
static inline void push_dynamic_array(DynamicIntArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (int*)realloc(arr->data, arr->capacity * sizeof(int));
    }
    arr->data[arr->size++] = value;
}

/**
 * @brief Free dynamic array memory
 * @param arr Pointer to DynamicIntArray structure
 */
static inline void free_dynamic_array(DynamicIntArray* arr) {
    if (arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// ============================================================================
// FASTA FILE PARSING
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse FASTA file and extract sequences
 * @param filename Path to FASTA file
 * @param num_sequences Output: number of sequences found
 * @return Array of sequence strings (NULL-terminated)
 * 
 * Parses a FASTA format file and returns an array of sequences.
 * Each sequence is a null-terminated string.
 * Caller is responsible for freeing the returned array and each sequence.
 * 
 * FASTA format:
 *   >Header line (ignored)
 *   SEQUENCE_DATA_LINE_1
 *   SEQUENCE_DATA_LINE_2
 *   ...
 *   >Next sequence header
 *   ...
 */
char** parse_fasta_file(const char *filename, int *num_sequences);

/**
 * @brief Read entire file into a string
 * @param filename Path to file
 * @return Pointer to file contents (NULL-terminated), or NULL on error
 * 
 * Caller is responsible for freeing the returned string.
 */
char* read_file_into_string(const char* filename);

#ifdef __cplusplus
}
#endif

// ============================================================================
// RESULT AGGREGATION
// ============================================================================

/**
 * @brief Aggregate and deduplicate indices from multiple chunks
 * @param chunk_indices Array of index arrays (one per chunk)
 * @param chunk_counts Array of counts (one per chunk)
 * @param num_chunks Number of chunks
 * @param output_indices Output: pointer to receive merged array
 * @param output_count Output: number of unique indices
 * 
 * Combines indices from multiple chunks, sorts them, and removes duplicates.
 * Allocates memory for output_indices which caller must free.
 */
static inline void aggregate_indices(
    int** chunk_indices, 
    int* chunk_counts, 
    int num_chunks,
    int** output_indices,
    int* output_count)
{
    // Count total indices
    int total = 0;
    for (int i = 0; i < num_chunks; i++) {
        total += chunk_counts[i];
    }
    
    if (total == 0) {
        *output_indices = NULL;
        *output_count = 0;
        return;
    }
    
    // Allocate temporary array
    int* temp = (int*)malloc(total * sizeof(int));
    int write_pos = 0;
    
    // Copy all indices
    for (int i = 0; i < num_chunks; i++) {
        for (int j = 0; j < chunk_counts[i]; j++) {
            temp[write_pos++] = chunk_indices[i][j];
        }
    }
    
    // Sort and deduplicate
    *output_count = write_pos;
    sort_and_deduplicate(temp, output_count);
    
    // Allocate final array with exact size
    *output_indices = (int*)malloc(*output_count * sizeof(int));
    memcpy(*output_indices, temp, *output_count * sizeof(int));
    
    free(temp);
}

/**
 * @brief Merge indices with offset adjustment
 * @param gpu_indices GPU indices array
 * @param gpu_count Number of GPU indices
 * @param fpga_indices FPGA indices array
 * @param fpga_count Number of FPGA indices
 * @param fpga_offset Offset to add to FPGA indices
 * @param output_indices Output: pointer to receive merged array
 * @param output_count Output: number of unique indices
 * 
 * Combines GPU and FPGA indices, adds offset to FPGA indices,
 * sorts and deduplicates the result.
 */
static inline void merge_indices_with_offset(
    int* gpu_indices,
    int gpu_count,
    unsigned long* fpga_indices,
    int fpga_count,
    int fpga_offset,
    int** output_indices,
    int* output_count)
{
    int total = gpu_count + fpga_count;
    if (total == 0) {
        *output_indices = NULL;
        *output_count = 0;
        return;
    }
    
    // Allocate temporary array
    int* temp = (int*)malloc(total * sizeof(int));
    int pos = 0;
    
    // Copy GPU indices
    for (int i = 0; i < gpu_count; i++) {
        temp[pos++] = gpu_indices[i];
    }
    
    // Copy FPGA indices with offset
    for (int i = 0; i < fpga_count; i++) {
        temp[pos++] = (int)fpga_indices[i] + fpga_offset;
    }
    
    // Sort and deduplicate
    *output_count = pos;
    sort_and_deduplicate(temp, output_count);
    
    // Allocate final array
    *output_indices = (int*)malloc(*output_count * sizeof(int));
    memcpy(*output_indices, temp, *output_count * sizeof(int));
    
    free(temp);
}

#endif // CPU_UTILS_H