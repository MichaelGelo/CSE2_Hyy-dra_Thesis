/**
 * @file hyrro_partition.h
 * @brief Reference sequence partitioning utilities
 * 
 * This file provides functions to partition long reference sequences into
 * smaller overlapping chunks. This is necessary because:
 * 1. GPU memory is limited
 * 2. Smaller chunks can be processed in parallel
 * 3. Overlaps ensure no matches are missed at chunk boundaries
 */

#ifndef HYRRO_PARTITION_H
#define HYRRO_PARTITION_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PARTITIONED REFERENCES STRUCTURE
// ============================================================================

/**
 * @brief Structure to hold partitioned reference data
 * 
 * After partitioning, long references are split into overlapping chunks.
 * This structure maintains the relationship between chunks and original refs.
 */
typedef struct {
    char** chunk_seqs;           ///< Array of chunk sequences
    int* chunk_lens;             ///< Length of each chunk
    int* chunk_starts;           ///< Start position in original reference
    int* chunk_to_orig;          ///< Maps chunk index to original reference index
    int num_chunks;              ///< Total number of chunks
} PartitionedRefs;

// ============================================================================
// PARTITIONING FUNCTION
// ============================================================================

/**
 * @brief Partition references into overlapping chunks
 * 
 * Long references (> partition_threshold) are split into chunks of size
 * chunk_size with overlap between consecutive chunks. The overlap ensures
 * that matches spanning chunk boundaries are not missed.
 * 
 * @param orig_refs Array of original reference sequences
 * @param orig_ref_lens Array of original reference lengths
 * @param num_orig_refs Number of original references
 * @param overlap Overlap size between chunks (typically query_length - 1)
 * @param chunk_size Size of each chunk
 * @param partition_threshold References longer than this will be partitioned
 * @return PartitionedRefs structure containing all chunk data
 * 
 * Example:
 *   Reference: "ABCDEFGHIJKLMNOP" (length 16)
 *   chunk_size = 6, overlap = 2
 *   
 *   Chunk 0: "ABCDEFGH"   (0-7, with 2-char overlap)
 *   Chunk 1: "GHIJKLMN"   (6-13, with 2-char overlap)
 *   Chunk 2: "MNOP"       (12-15, last chunk)
 *   
 *   Note: Each chunk includes overlap with next chunk to ensure
 *         no matches are missed at boundaries.
 */
static inline PartitionedRefs partition_references(
    char** orig_refs,
    int* orig_ref_lens,
    int num_orig_refs,
    int overlap,
    int chunk_size,
    int partition_threshold)
{
    PartitionedRefs result;
    result.num_chunks = 0;

    // ========== PASS 1: Count total chunks needed ==========
    int estimated_chunks = 0;
    for (int i = 0; i < num_orig_refs; ++i) {
        if (orig_ref_lens[i] > partition_threshold) {
            // Calculate number of chunks for this reference
            int nch = (orig_ref_lens[i] + chunk_size - 1) / chunk_size;
            estimated_chunks += nch;
        } else {
            // Small reference = single chunk
            estimated_chunks += 1;
        }
    }

    // ========== Allocate arrays for chunks ==========
    result.chunk_seqs = (char**)malloc(sizeof(char*) * estimated_chunks);
    result.chunk_lens = (int*)malloc(sizeof(int) * estimated_chunks);
    result.chunk_starts = (int*)malloc(sizeof(int) * estimated_chunks);
    result.chunk_to_orig = (int*)malloc(sizeof(int) * estimated_chunks);

    if (!result.chunk_seqs || !result.chunk_lens || 
        !result.chunk_starts || !result.chunk_to_orig) {
        fprintf(stderr, "Failed to allocate partition arrays\n");
        exit(EXIT_FAILURE);
    }

    // ========== PASS 2: Create chunks ==========
    int chunk_idx = 0;
    
    for (int r = 0; r < num_orig_refs; ++r) {
        int rlen = orig_ref_lens[r];

        if (rlen > partition_threshold) {
            // === Partition this reference into chunks with overlap ===
            int nch = (rlen + chunk_size - 1) / chunk_size;
            
            for (int c = 0; c < nch; ++c) {
                int start = c * chunk_size;
                int len = chunk_size;
                
                // Adjust length for last chunk
                if (start + len > rlen) {
                    len = rlen - start;
                }

                // Add overlap (typically query_length - 1)
                int ext_len = len + overlap;
                if (start + ext_len > rlen) {
                    ext_len = rlen - start;
                }

                // Allocate and copy chunk
                char* s = (char*)malloc(ext_len + 1);
                if (!s) {
                    fprintf(stderr, "Failed to allocate chunk\n");
                    exit(EXIT_FAILURE);
                }
                
                memcpy(s, orig_refs[r] + start, ext_len);
                s[ext_len] = '\0';

                // Store chunk metadata
                result.chunk_seqs[chunk_idx] = s;
                result.chunk_lens[chunk_idx] = ext_len;
                result.chunk_starts[chunk_idx] = start;
                result.chunk_to_orig[chunk_idx] = r;
                chunk_idx++;
            }
        } else {
            // === Reference is small, use as single chunk ===
            int ext_len = rlen;
            char* s = (char*)malloc(ext_len + 1);
            
            if (!s) {
                fprintf(stderr, "Failed to allocate chunk\n");
                exit(EXIT_FAILURE);
            }
            
            memcpy(s, orig_refs[r], ext_len);
            s[ext_len] = '\0';

            result.chunk_seqs[chunk_idx] = s;
            result.chunk_lens[chunk_idx] = ext_len;
            result.chunk_starts[chunk_idx] = 0;
            result.chunk_to_orig[chunk_idx] = r;
            chunk_idx++;
        }
    }

    result.num_chunks = chunk_idx;
    return result;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Free memory allocated for partitioned references
 * @param part_refs Pointer to PartitionedRefs structure
 * 
 * Frees all chunk sequences and metadata arrays.
 */
static inline void free_partitioned_refs(PartitionedRefs* part_refs) {
    if (!part_refs) return;
    
    // Free each chunk sequence
    if (part_refs->chunk_seqs) {
        for (int i = 0; i < part_refs->num_chunks; ++i) {
            if (part_refs->chunk_seqs[i]) {
                free(part_refs->chunk_seqs[i]);
            }
        }
        free(part_refs->chunk_seqs);
    }
    
    // Free metadata arrays
    if (part_refs->chunk_lens) free(part_refs->chunk_lens);
    if (part_refs->chunk_starts) free(part_refs->chunk_starts);
    if (part_refs->chunk_to_orig) free(part_refs->chunk_to_orig);
    
    // Reset structure
    part_refs->chunk_seqs = NULL;
    part_refs->chunk_lens = NULL;
    part_refs->chunk_starts = NULL;
    part_refs->chunk_to_orig = NULL;
    part_refs->num_chunks = 0;
}

// ============================================================================
// CHUNK-TO-ORIGINAL MAPPING
// ============================================================================

/**
 * @brief Build mapping of original references to their chunks
 * 
 * Creates reverse mapping to quickly find all chunks belonging to
 * each original reference. Useful for result aggregation.
 * 
 * @param part_refs Partitioned reference structure
 * @param num_orig_refs Number of original references
 * @param out_counts Output: array of chunk counts per original ref
 * @param out_lists Output: array of chunk lists per original ref
 * 
 * Example:
 *   Original ref 0 has 3 chunks: [0, 1, 2]
 *   Original ref 1 has 1 chunk:  [3]
 *   Original ref 2 has 2 chunks: [4, 5]
 *   
 *   out_counts = [3, 1, 2]
 *   out_lists[0] = [0, 1, 2]
 *   out_lists[1] = [3]
 *   out_lists[2] = [4, 5]
 */
static inline void build_orig_to_chunk_mapping(
    const PartitionedRefs* part_refs,
    int num_orig_refs,
    int** out_counts,
    int*** out_lists)
{
    // ========== Allocate and initialize counts ==========
    int* orig_chunk_counts = (int*)calloc(num_orig_refs, sizeof(int));
    
    if (!orig_chunk_counts) {
        fprintf(stderr, "Failed to allocate chunk counts\n");
        exit(EXIT_FAILURE);
    }

    // ========== Count chunks per original reference ==========
    for (int r = 0; r < part_refs->num_chunks; ++r) {
        orig_chunk_counts[part_refs->chunk_to_orig[r]]++;
    }

    // ========== Allocate chunk lists ==========
    int** orig_chunk_lists = (int**)malloc(sizeof(int*) * num_orig_refs);
    
    if (!orig_chunk_lists) {
        fprintf(stderr, "Failed to allocate chunk lists\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < num_orig_refs; ++i) {
        if (orig_chunk_counts[i] > 0) {
            orig_chunk_lists[i] = (int*)malloc(sizeof(int) * orig_chunk_counts[i]);
            if (!orig_chunk_lists[i]) {
                fprintf(stderr, "Failed to allocate chunk list\n");
                exit(EXIT_FAILURE);
            }
        } else {
            orig_chunk_lists[i] = NULL;
        }
        
        // Reset count for second pass
        orig_chunk_counts[i] = 0;
    }

    // ========== Fill chunk lists ==========
    for (int r = 0; r < part_refs->num_chunks; ++r) {
        int o = part_refs->chunk_to_orig[r];
        orig_chunk_lists[o][orig_chunk_counts[o]++] = r;
    }

    *out_counts = orig_chunk_counts;
    *out_lists = orig_chunk_lists;
}

/**
 * @brief Free orig-to-chunk mapping
 * @param counts Array of chunk counts
 * @param lists Array of chunk lists
 * @param num_orig_refs Number of original references
 */
static inline void free_orig_to_chunk_mapping(int* counts, int** lists, int num_orig_refs) {
    if (counts) free(counts);
    
    if (lists) {
        for (int i = 0; i < num_orig_refs; ++i) {
            if (lists[i]) free(lists[i]);
        }
        free(lists);
    }
}

#ifdef __cplusplus
}
#endif

#endif // HYRRO_PARTITION_H