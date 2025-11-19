// output_processor.h
// Processing and printing of computation results

#ifndef OUTPUT_PROCESSOR_H
#define OUTPUT_PROCESSOR_H

#include <stdlib.h>
#include <limits.h>
#include "config.h"

// ============================================================================
// INTEGER COMPARISON FOR QSORT
// Used for sorting hit indices and removing duplicates
// ============================================================================
static inline int compareInts(const void *a, const void *b) {
    int av = *(const int*)a;
    int bv = *(const int*)b;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
}

// ============================================================================
// REMOVE DUPLICATE INDICES
// Sorts array and removes consecutive duplicates, returns new count
// ============================================================================
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
// PRINT ARRAY OF INTEGERS
// Formats and prints an array in bracket notation: [1,2,3]
// ============================================================================
static inline void printIntArray(const int* arr, size_t count) {
    printf("[");
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) printf(",");
        printf("%d", arr[i]);
    }
    printf("]");
}

// ============================================================================
// COLLECT ZERO-SCORE HITS FOR ORIGINAL REFERENCE
// Aggregates all exact match positions across chunks belonging to one reference
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

// ============================================================================
// COLLECT AND DEDUPLICATE LOWEST SCORE INDICES
// Extracts valid indices and removes duplicates
// ============================================================================
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
// PRINT RESULTS FOR ONE QUERY-REFERENCE PAIR
// Displays all computed metrics in a formatted block
// ============================================================================
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

#endif // OUTPUT_PROCESSOR_H