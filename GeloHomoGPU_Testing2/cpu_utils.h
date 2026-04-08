// cpu_utils.h
// CPU-side utilities: file I/O, result processing, and output formatting

#ifndef HYRRO_CPU_H
#define HYRRO_CPU_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// RESULT DATA STRUCTURE FOR CSV EXPORT
// ============================================================================
typedef struct {
    char query_filename[512];
    int query_length;
    char reference_filename[512];
    int reference_length;
    int num_hits;
    int* hit_indexes;
    int lowest_score;
    int* lowest_score_indexes;
    int lowest_score_count;
    int last_score;
} PairResult;

// ============================================================================
// FILE I/O - FASTA PARSING
// ============================================================================

#define MAX_LINE_LENGTH (1 << 14)

char* read_file_into_string(const char* filename);
char** parse_fasta_file(const char *filename, int *num_sequences);

// ============================================================================
// DATA LOADING - QUERIES AND REFERENCES (FOLDER-BASED)
// ============================================================================
// These functions are now in homocuda.cu for multi-file support
// See: loadQueriesFromFolder() and loadReferencesFromFolder()

// Legacy single-file loading (kept for backward compatibility):
// static inline char** loadQueries(int *numQueries) {
//     return parse_fasta_file(QUERY_FILE, numQueries);
// }
// 
// static inline char** loadReferences(int *numRefs, int **refLens) {
//     int num = 0;
//     char **refs = parse_fasta_file(REFERENCE_FILE, &num);
//     ...
// }

static inline int* computeQueryLengths(char** queries, int numQueries) {
    int* lengths = (int*)malloc(numQueries * sizeof(int));
    if (!lengths) {
        fprintf(stderr, "ERROR: Out of memory for query lengths\n");
        return NULL;
    }
    
    for (int q = 0; q < numQueries; ++q) {
        lengths[q] = (int)strlen(queries[q]);
    }
    
    return lengths;
}

// ============================================================================
// RESULT PROCESSING - SORTING AND DEDUPLICATION
// ============================================================================

static inline int compareInts(const void *a, const void *b) {
    int av = *(const int*)a;
    int bv = *(const int*)b;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
}

static inline size_t removeDuplicates(int* arr, size_t count) {
    if (count == 0) return 0;
    
    qsort(arr, count, sizeof(int), compareInts);
    
    size_t write = 1;
    for (size_t i = 1; i < count; ++i) {
        if (arr[i] != arr[write - 1]) {
            arr[write++] = arr[i];
        }
    }
    
    return write;
}

// ============================================================================
// RESULT COLLECTION - ZERO HITS AND LOWEST SCORES
// ============================================================================

static inline int* collectZeroHits(
    int q, int orig, int numChunks,
    int* origChunkCounts, int** origChunkLists,
    int* hostPairZcounts, int* hostPairZindices,
    size_t* outCount)
{
    size_t capacity = 4096;
    int* hits = (int*)malloc(sizeof(int) * capacity);
    size_t count = 0;
    
    for (int ci = 0; ci < origChunkCounts[orig]; ++ci) {
        int chunkId = origChunkLists[orig][ci];
        long long pairIdx = (long long)q * numChunks + chunkId;
        int zeroCount = hostPairZcounts[pairIdx];
        
        for (int k = 0; k < zeroCount && k < MAX_HITS; ++k) {
            int val = hostPairZindices[pairIdx * MAX_HITS + k];
            if (val >= 0) {
                if (count >= capacity) {
                    capacity *= 2;
                    hits = (int*)realloc(hits, sizeof(int) * capacity);
                }
                hits[count++] = val;
            }
        }
    }
    
    if (count > 0) {
        count = removeDuplicates(hits, count);
    }
    
    *outCount = count;
    return hits;
}

static inline int* collectLowestIndices(
    int* rawIndices, int rawCount, size_t* outCount)
{
    if (rawCount <= 0) {
        *outCount = 0;
        return NULL;
    }
    
    int* indices = (int*)malloc(sizeof(int) * MIN(rawCount, MAX_HITS));
    int validCount = 0;
    
    for (int k = 0; k < MIN(rawCount, MAX_HITS); ++k) {
        if (rawIndices[k] >= 0) {
            indices[validCount++] = rawIndices[k];
        }
    }
    
    if (validCount > 0) {
        validCount = removeDuplicates(indices, validCount);
    }
    
    *outCount = validCount;
    return indices;
}

// ============================================================================
// OUTPUT FORMATTING
// ============================================================================

static inline void printIntArray(const int* arr, size_t count) {
    printf("[");
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) printf(",");
        printf("%d", arr[i]);
    }
    printf("]");
}

static inline void printPairResults(
    int q, int orig,
    int queryLength, int refLength,
    size_t zeroHitCount, int* zeroHits,
    int lowestScore, size_t lowestCount, int* lowestIndices,
    int lastScore)
{
    printf("----------------------------------------------------------------------------\n");
    printf("Pair: Q%d(%d) Vs R%d(%d)\n", q + 1, queryLength, orig + 1, refLength);
    
    printf("Number of Hits: %zu\n", zeroHitCount);
    if (zeroHitCount > 0) {
        printf("Hit Indexes: ");
        printIntArray(zeroHits, zeroHitCount);
        printf("\n");
    } else {
        printf("Hit Indexes: N/A\n");
    }
    
    if (lowestScore == 0x7f7f7f7f) {
        printf("Lowest Score: N/A\n");
        printf("Lowest Score Indexes: N/A\n");
    } else {
        printf("Lowest Score: %d\n", lowestScore);
        if (lowestCount > 0) {
            printf("Lowest Score Indexes: ");
            printIntArray(lowestIndices, lowestCount);
            printf("\n");
        } else {
            printf("Lowest Score Indexes: N/A\n");
        }
    }
    
    if (lastScore == 0x7f7f7f7f) {
        printf("Last Score: N/A\n");
    } else {
        printf("Last Score: %d\n", lastScore);
    }
    
    printf("----------------------------------------------------------------------------\n");
}

// ============================================================================
// CSV EXPORT FUNCTIONS
// ============================================================================

static inline void write_csv_header(FILE* csv_file) {
    fprintf(csv_file, "Homo GPU Run\n");
    fprintf(csv_file, "Query File,Query Length,Reference File,Reference Length,"
                      "Number of Hits,Hit Indexes,Lowest Score,Lowest Score Indexes,Last Score\n");
}

static inline void write_int_array_csv(FILE* csv_file, const int* arr, int count) {
    fprintf(csv_file, "\"");
    for (int i = 0; i < count; ++i) {
        if (i > 0) fprintf(csv_file, ",");
        fprintf(csv_file, "%d", arr[i]);
    }
    fprintf(csv_file, "\"");
}

static inline void write_pair_result_csv(FILE* csv_file, const PairResult* result) {
    // Query filename (use basename only)
    const char* query_name = strrchr(result->query_filename, '/');
    query_name = query_name ? query_name + 1 : result->query_filename;
    
    // Reference filename (use basename only)
    const char* ref_name = strrchr(result->reference_filename, '/');
    ref_name = ref_name ? ref_name + 1 : result->reference_filename;
    
    fprintf(csv_file, "%s,%d,%s,%d,%d,",
            query_name, result->query_length,
            ref_name, result->reference_length,
            result->num_hits);
    
    // Hit indexes
    if (result->num_hits > 0 && result->hit_indexes) {
        write_int_array_csv(csv_file, result->hit_indexes, result->num_hits);
    } else {
        fprintf(csv_file, "\"N/A\"");
    }
    
    fprintf(csv_file, ",");
    
    // Lowest score
    if (result->lowest_score == 0x7f7f7f7f) {
        fprintf(csv_file, "N/A,\"N/A\"");
    } else {
        fprintf(csv_file, "%d,", result->lowest_score);
        // Lowest score indexes
        if (result->lowest_score_count > 0 && result->lowest_score_indexes) {
            write_int_array_csv(csv_file, result->lowest_score_indexes, result->lowest_score_count);
        } else {
            fprintf(csv_file, "\"N/A\"");
        }
    }
    
    fprintf(csv_file, ",");
    
    // Last score
    if (result->last_score == 0x7f7f7f7f) {
        fprintf(csv_file, "N/A");
    } else {
        fprintf(csv_file, "%d", result->last_score);
    }
    
    fprintf(csv_file, "\n");
}

#ifdef __cplusplus
}
#endif

#endif // HYYRO_CPU_H