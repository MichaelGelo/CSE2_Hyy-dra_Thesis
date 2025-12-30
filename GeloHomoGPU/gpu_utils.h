// gpu_utils.h
// GPU utilities: error checking, timing, and memory management

#ifndef HYRRO_GPU_H
#define HYRRO_GPU_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "bitvector.h"
#include "hyrro_partition.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================================================
// HIGH-PRECISION TIMING
// ============================================================================
static inline double getNowSeconds() {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER count;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

// ============================================================================
// GPU MEMORY BUFFERS
// ============================================================================
typedef struct {
    // Input data
    bv_t* d_EqQueries;
    char* d_queries;
    char* d_refs;
    int* d_qLens;
    int* d_refLens;
    
    // Partitioning metadata
    int* d_chunkStarts;
    int* d_chunkToOrig;
    int* d_origRefLens;
    
    // Per-chunk output
    int* d_pairDistances;
    int* d_pairZcounts;
    int* d_pairZindices;
    
    // Aggregated output
    int* d_lowestScoreOrig;
    int* d_lowestCountOrig;
    int* d_lowestIndicesOrig;
    int* d_lastScoreOrig;
} GpuBuffers;

// ============================================================================
// ALLOCATE GPU MEMORY
// ============================================================================
static inline bool allocateGpuMemory(
    GpuBuffers* buffers,
    int numQueries,
    int numChunks,
    int numOrigRefs)
{
    long long totalPairChunks = (long long)numQueries * numChunks;
    long long totalOrigPairs = (long long)numQueries * numOrigRefs;
    
    // Input data
    CUDA_CHECK(cudaMalloc(&buffers->d_EqQueries, (size_t)numQueries * 256 * sizeof(bv_t)));
    CUDA_CHECK(cudaMalloc(&buffers->d_queries, (size_t)numQueries * MAX_LENGTH * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&buffers->d_refs, (size_t)numChunks * MAX_LENGTH * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&buffers->d_qLens, numQueries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_refLens, numChunks * sizeof(int)));
    
    // Partitioning metadata
    CUDA_CHECK(cudaMalloc(&buffers->d_chunkStarts, numChunks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_chunkToOrig, numChunks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_origRefLens, numOrigRefs * sizeof(int)));
    
    // Per-chunk output
    CUDA_CHECK(cudaMalloc(&buffers->d_pairDistances, totalPairChunks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_pairZcounts, totalPairChunks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_pairZindices, totalPairChunks * MAX_HITS * sizeof(int)));
    
    // Aggregated output
    CUDA_CHECK(cudaMalloc(&buffers->d_lowestScoreOrig, totalOrigPairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_lowestCountOrig, totalOrigPairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_lowestIndicesOrig, totalOrigPairs * MAX_HITS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers->d_lastScoreOrig, totalOrigPairs * sizeof(int)));
    
    return true;
}

// ============================================================================
// PACK SEQUENCES INTO CONTIGUOUS BUFFERS
// ============================================================================
static inline void packSequences(
    char** queries, int numQueries,
    char** refChunks, int numChunks,
    char** hostQueries, char** hostRefs)
{
    size_t queriesBytes = (size_t)numQueries * MAX_LENGTH * sizeof(char);
    size_t refsBytes = (size_t)numChunks * MAX_LENGTH * sizeof(char);
    
    memset(*hostQueries, 0, queriesBytes);
    memset(*hostRefs, 0, refsBytes);
    
    for (int q = 0; q < numQueries; ++q) {
        strncpy(&(*hostQueries)[(size_t)q * MAX_LENGTH],
                queries[q],
                MAX_LENGTH - 1);
    }
    
    for (int r = 0; r < numChunks; ++r) {
        strncpy(&(*hostRefs)[(size_t)r * MAX_LENGTH],
                refChunks[r],
                MAX_LENGTH - 1);
    }
}

// ============================================================================
// TRANSFER DATA TO GPU
// ============================================================================
static inline void transferToGpu(
    GpuBuffers* buffers,
    bv_t* hostEqTables,
    char* hostQueries,
    char* hostRefs,
    int* hostQLens,
    int* hostRefLens,
    PartitionedRefs* partRefs,
    int* origRefLens,
    int numQueries,
    int numChunks,
    int numOrigRefs)
{
    long long totalPairChunks = (long long)numQueries * numChunks;
    long long totalOrigPairs = (long long)numQueries * numOrigRefs;
    
    // Copy input data
    CUDA_CHECK(cudaMemcpy(buffers->d_EqQueries, hostEqTables,
                         (size_t)numQueries * 256 * sizeof(bv_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_queries, hostQueries,
                         (size_t)numQueries * MAX_LENGTH * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_refs, hostRefs,
                         (size_t)numChunks * MAX_LENGTH * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_qLens, hostQLens,
                         numQueries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_refLens, hostRefLens,
                         numChunks * sizeof(int), cudaMemcpyHostToDevice));
    
    // Copy partitioning metadata
    CUDA_CHECK(cudaMemcpy(buffers->d_chunkStarts, partRefs->chunk_starts,
                         numChunks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_chunkToOrig, partRefs->chunk_to_orig,
                         numChunks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_origRefLens, origRefLens,
                         numOrigRefs * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize output buffers
    CUDA_CHECK(cudaMemset(buffers->d_lowestScoreOrig, 0x7f, totalOrigPairs * sizeof(int)));
    CUDA_CHECK(cudaMemset(buffers->d_lowestCountOrig, 0, totalOrigPairs * sizeof(int)));
    CUDA_CHECK(cudaMemset(buffers->d_lowestIndicesOrig, 0xff, totalOrigPairs * MAX_HITS * sizeof(int)));
    CUDA_CHECK(cudaMemset(buffers->d_lastScoreOrig, 0x7f, totalOrigPairs * sizeof(int)));
    CUDA_CHECK(cudaMemset(buffers->d_pairDistances, 0xff, totalPairChunks * sizeof(int)));
    CUDA_CHECK(cudaMemset(buffers->d_pairZcounts, 0, totalPairChunks * sizeof(int)));
    CUDA_CHECK(cudaMemset(buffers->d_pairZindices, 0xff, totalPairChunks * MAX_HITS * sizeof(int)));
}

// ============================================================================
// FREE GPU MEMORY
// ============================================================================
static inline void freeGpuMemory(GpuBuffers* buffers) {
    CUDA_CHECK(cudaFree(buffers->d_EqQueries));
    CUDA_CHECK(cudaFree(buffers->d_queries));
    CUDA_CHECK(cudaFree(buffers->d_refs));
    CUDA_CHECK(cudaFree(buffers->d_qLens));
    CUDA_CHECK(cudaFree(buffers->d_refLens));
    CUDA_CHECK(cudaFree(buffers->d_chunkStarts));
    CUDA_CHECK(cudaFree(buffers->d_chunkToOrig));
    CUDA_CHECK(cudaFree(buffers->d_origRefLens));
    CUDA_CHECK(cudaFree(buffers->d_pairDistances));
    CUDA_CHECK(cudaFree(buffers->d_pairZcounts));
    CUDA_CHECK(cudaFree(buffers->d_pairZindices));
    CUDA_CHECK(cudaFree(buffers->d_lowestScoreOrig));
    CUDA_CHECK(cudaFree(buffers->d_lowestCountOrig));
    CUDA_CHECK(cudaFree(buffers->d_lowestIndicesOrig));
    CUDA_CHECK(cudaFree(buffers->d_lastScoreOrig));
}

#endif // HYRRO_GPU_H