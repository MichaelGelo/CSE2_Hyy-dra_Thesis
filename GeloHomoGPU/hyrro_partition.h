// hyrro_partition.h
// Reference sequence partitioning with overlap support

#ifndef HYRRO_PARTITION_H
#define HYRRO_PARTITION_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PARTITIONED REFERENCES STRUCTURE
// Holds all data for reference sequences split into overlapping chunks
// ============================================================================
typedef struct {
    char** chunk_seqs;      // Array of chunk sequences
    int* chunk_lens;        // Length of each chunk
    int* chunk_starts;      // Start position in original reference
    int* chunk_to_orig;     // Maps chunk index to original reference index
    int num_chunks;         // Total number of chunks
} PartitionedRefs;

// ============================================================================
// PARTITION REFERENCES INTO OVERLAPPING CHUNKS
// Splits long references into smaller chunks with overlap for boundary handling
// ============================================================================
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

    // First pass: count total chunks needed
    int estimated_chunks = 0;
    for (int i = 0; i < num_orig_refs; ++i) {
        if (orig_ref_lens[i] > partition_threshold) {
            int nch = (orig_ref_lens[i] + chunk_size - 1) / chunk_size;
            estimated_chunks += nch;
        } else {
            estimated_chunks += 1;
        }
    }

    // Allocate arrays for chunks
    result.chunk_seqs = (char**)malloc(sizeof(char*) * estimated_chunks);
    result.chunk_lens = (int*)malloc(sizeof(int) * estimated_chunks);
    result.chunk_starts = (int*)malloc(sizeof(int) * estimated_chunks);
    result.chunk_to_orig = (int*)malloc(sizeof(int) * estimated_chunks);

    // Second pass: create chunks
    int chunk_idx = 0;
    for (int r = 0; r < num_orig_refs; ++r) {
        int rlen = orig_ref_lens[r];

        if (rlen > partition_threshold) {
            // Partition this reference into chunks with overlap
            int nch = (rlen + chunk_size - 1) / chunk_size;
            for (int c = 0; c < nch; ++c) {
                int start = c * chunk_size;
                int len = chunk_size;
                if (start + len > rlen) len = rlen - start;

                // Add overlap (typically query_length - 1)
                int ext_len = len + overlap;
                if (start + ext_len > rlen) ext_len = rlen - start;

                // Allocate and copy chunk
                char* s = (char*)malloc(ext_len + 1);
                memcpy(s, orig_refs[r] + start, ext_len);
                s[ext_len] = '\0';

                result.chunk_seqs[chunk_idx] = s;
                result.chunk_lens[chunk_idx] = ext_len;
                result.chunk_starts[chunk_idx] = start;
                result.chunk_to_orig[chunk_idx] = r;
                chunk_idx++;
            }
        } else {
            // Reference is small, use as single chunk
            int ext_len = rlen;
            char* s = (char*)malloc(ext_len + 1);
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
// FREE PARTITIONED REFERENCES
// Releases all memory allocated for partitioned reference data
// ============================================================================
static inline void free_partitioned_refs(PartitionedRefs* part_refs) {
    if (part_refs->chunk_seqs) {
        for (int i = 0; i < part_refs->num_chunks; ++i) {
            if (part_refs->chunk_seqs[i]) free(part_refs->chunk_seqs[i]);
        }
        free(part_refs->chunk_seqs);
    }
    if (part_refs->chunk_lens) free(part_refs->chunk_lens);
    if (part_refs->chunk_starts) free(part_refs->chunk_starts);
    if (part_refs->chunk_to_orig) free(part_refs->chunk_to_orig);
}

// ============================================================================
// BUILD MAPPING OF ORIGINAL REFERENCES TO THEIR CHUNKS
// Creates reverse lookup: for each original reference, lists all its chunks
// ============================================================================
static inline void build_orig_to_chunk_mapping(
    const PartitionedRefs* part_refs,
    int num_orig_refs,
    int** out_counts,
    int*** out_lists)
{
    // Allocate and initialize counts
    int* orig_chunk_counts = (int*)calloc(num_orig_refs, sizeof(int));

    // Count chunks per original reference
    for (int r = 0; r < part_refs->num_chunks; ++r) {
        orig_chunk_counts[part_refs->chunk_to_orig[r]]++;
    }

    // Allocate lists
    int** orig_chunk_lists = (int**)malloc(sizeof(int*) * num_orig_refs);
    for (int i = 0; i < num_orig_refs; ++i) {
        if (orig_chunk_counts[i] > 0) {
            orig_chunk_lists[i] = (int*)malloc(sizeof(int) * orig_chunk_counts[i]);
        } else {
            orig_chunk_lists[i] = NULL;
        }
        orig_chunk_counts[i] = 0; // Reset for second pass
    }

    // Fill lists
    for (int r = 0; r < part_refs->num_chunks; ++r) {
        int o = part_refs->chunk_to_orig[r];
        orig_chunk_lists[o][orig_chunk_counts[o]++] = r;
    }

    *out_counts = orig_chunk_counts;
    *out_lists = orig_chunk_lists;
}

// ============================================================================
// FREE CHUNK MAPPING
// Releases memory allocated for original-to-chunk mapping
// ============================================================================
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

#endif // HYYRO_PARTITION_H