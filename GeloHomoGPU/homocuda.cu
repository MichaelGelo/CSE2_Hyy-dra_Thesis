// homocuda.cu
// Main entry point for Hyyrö bit-vector Levenshtein distance computation
// WITH DYNAMIC PARTITIONING based on workload
// Compile: nvcc -O3 homocuda.cu -o levenshtein_gpu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "hyrro_partition.h"
#include "bitvector.h"
#include "cpu_utils.h"
#include "hyrro_io.h"
#include "gpu_utils.h"
#include "levenshtein_kernel.cuh"

// ============================================================================
// RUN HYYRÖ ALGORITHM ON GPU
// ============================================================================
void runHyyroAlgorithm(
    int numQueries, int numChunks, int numOrigRefs,
    char** querySeqs, char** origRefs,
    int* origRefLens, PartitionedRefs* partRefs,
    int* origChunkCounts, int** origChunkLists,
    int* queryLengths)
{
    printf("\n=== Running Hyyrö GPU Pipeline ===\n");

    // Build Eq tables
    bv_t* hostEqTables = buildEqTables(querySeqs, queryLengths, numQueries);
    if (!hostEqTables) {
        fprintf(stderr, "ERROR: Failed to build Eq tables\n");
        return;
    }

    // Pack sequences EFFICIENTLY (no padding waste)
    char* hostQueries = NULL;
    char* hostRefs = NULL;
    int* hostRefOffsets = NULL;
    size_t totalRefBytes = 0;
    
    packSequencesEfficient(querySeqs, numQueries, 
                          partRefs->chunk_seqs, partRefs->chunk_lens, numChunks,
                          &hostQueries, &hostRefs, &hostRefOffsets, &totalRefBytes);
    
    if (!hostQueries || !hostRefs || !hostRefOffsets) {
        fprintf(stderr, "ERROR: Out of memory for host buffers\n");
        free(hostEqTables);
        return;
    }

    // Allocate GPU memory with ACTUAL sizes
    GpuBuffers gpuBuffers;
    allocateGpuMemory(&gpuBuffers, numQueries, numChunks, numOrigRefs, totalRefBytes);

    // Transfer data to GPU
    transferToGpuEfficient(&gpuBuffers, hostEqTables, hostQueries, hostRefs,
                          queryLengths, partRefs->chunk_lens, hostRefOffsets,
                          partRefs, origRefLens,
                          numQueries, numChunks, numOrigRefs, totalRefBytes);

    // Launch optimized kernel: one block per query-chunk pair
    int totalPairs = numQueries * numChunks;
    int threads = THREADS_PER_BLOCK;
    int blocks = totalPairs;
    size_t sharedBytes = 256 * sizeof(bv_t);

    printf("*** OPTIMIZED Kernel launch: %d blocks × %d threads, shared=%zu bytes\n",
           blocks, threads, sharedBytes);
    printf("    Total pairs: %d queries × %d chunks = %d blocks\n",
           numQueries, numChunks, totalPairs);
    printf("    GPU utilization: %d parallel computations\n", blocks);

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    float total_ms = 0.0f;

    for (int it = 0; it < LOOP_ITERATIONS; ++it) {

        // Start timing Hyyrö only
        CUDA_CHECK(cudaEventRecord(ev_start));

        levenshteinKernelOptimized<<<blocks, threads, sharedBytes>>>(
            numQueries, numChunks, numOrigRefs,
            gpuBuffers.d_queries, gpuBuffers.d_qLens, gpuBuffers.d_EqQueries,
            gpuBuffers.d_refs, gpuBuffers.d_refLens, gpuBuffers.d_refOffsets,
            gpuBuffers.d_chunkStarts, gpuBuffers.d_chunkToOrig,
            gpuBuffers.d_origRefLens,
            gpuBuffers.d_pairDistances, gpuBuffers.d_pairZcounts,
            gpuBuffers.d_pairZindices,
            gpuBuffers.d_lowestScoreOrig, gpuBuffers.d_lowestCountOrig,
            gpuBuffers.d_lowestIndicesOrig, gpuBuffers.d_lastScoreOrig
        );

        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));

        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_start, ev_end));
        total_ms += iter_ms;
    }

    // Run collect kernel outside timing
    collectLowestIndicesKernel<<<blocks, threads, sharedBytes>>>(
        numQueries, numChunks, numOrigRefs,
        gpuBuffers.d_queries, gpuBuffers.d_qLens, gpuBuffers.d_EqQueries,
        gpuBuffers.d_refs, gpuBuffers.d_refLens, gpuBuffers.d_refOffsets,
        gpuBuffers.d_chunkStarts, gpuBuffers.d_chunkToOrig,
        gpuBuffers.d_lowestScoreOrig, gpuBuffers.d_lowestCountOrig,
        gpuBuffers.d_lowestIndicesOrig
    );


    double avgTime = (total_ms / LOOP_ITERATIONS) / 1000.0; // seconds

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    // Copy results back to host
    long long totalPairChunks = (long long)numQueries * numChunks;
    long long totalOrigPairs = (long long)numQueries * numOrigRefs;
    
    int* hostPairDistances = (int*)malloc(totalPairChunks * sizeof(int));
    int* hostPairZcounts = (int*)malloc(totalPairChunks * sizeof(int));
    int* hostPairZindices = (int*)malloc(totalPairChunks * MAX_HITS * sizeof(int));
    int* hostLowestScoreOrig = (int*)malloc(totalOrigPairs * sizeof(int));
    int* hostLowestCountOrig = (int*)malloc(totalOrigPairs * sizeof(int));
    int* hostLowestIndicesOrig = (int*)malloc(totalOrigPairs * MAX_HITS * sizeof(int));
    int* hostLastScoreOrig = (int*)malloc(totalOrigPairs * sizeof(int));

    CUDA_CHECK(cudaMemcpy(hostPairDistances, gpuBuffers.d_pairDistances,
                         totalPairChunks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostPairZcounts, gpuBuffers.d_pairZcounts,
                         totalPairChunks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostPairZindices, gpuBuffers.d_pairZindices,
                         totalPairChunks * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostLowestScoreOrig, gpuBuffers.d_lowestScoreOrig,
                         totalOrigPairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostLowestCountOrig, gpuBuffers.d_lowestCountOrig,
                         totalOrigPairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostLowestIndicesOrig, gpuBuffers.d_lowestIndicesOrig,
                         totalOrigPairs * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostLastScoreOrig, gpuBuffers.d_lastScoreOrig,
                         totalOrigPairs * sizeof(int), cudaMemcpyDeviceToHost));

    // Process and print results
    for (int orig = 0; orig < numOrigRefs; ++orig) {
        for (int q = 0; q < numQueries; ++q) {
            // Collect zero-score hits
            size_t zeroHitCount = 0;
            int* zeroHits = collectZeroHits(q, orig, numChunks,
                                           origChunkCounts, origChunkLists,
                                           hostPairZcounts, hostPairZindices,
                                           &zeroHitCount);

            // Get lowest score info
            long long origPairIdx = (long long)q * numOrigRefs + orig;
            int lowestScore = hostLowestScoreOrig[origPairIdx];
            int lowestRawCount = hostLowestCountOrig[origPairIdx];
            
            size_t lowestCount = 0;
            int* lowestIndices = NULL;
            
            if (lowestScore != 0x7f7f7f7f && lowestRawCount > 0) {
                long long indicesBase = origPairIdx * MAX_HITS;
                lowestIndices = collectLowestIndices(
                    &hostLowestIndicesOrig[indicesBase],
                    lowestRawCount, &lowestCount);
            }

            int lastScore = hostLastScoreOrig[origPairIdx];

            // Print results
            printPairResults(q, orig, queryLengths[q], origRefLens[orig],
                           zeroHitCount, zeroHits,
                           lowestScore, lowestCount, lowestIndices,
                           lastScore);

            free(zeroHits);
            if (lowestIndices) free(lowestIndices);
        }
    }

    printf("\n%d loop Average time: %.6f sec.\n", LOOP_ITERATIONS, avgTime);

    // Cleanup
    freeGpuMemory(&gpuBuffers);
    free(hostEqTables);
    free(hostQueries);
    free(hostRefs);
    free(hostRefOffsets);
    free(hostPairDistances);
    free(hostPairZcounts);
    free(hostPairZindices);
    free(hostLowestScoreOrig);
    free(hostLowestCountOrig);
    free(hostLowestIndicesOrig);
    free(hostLastScoreOrig);
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
int main() {
    printf("=== Hyyrö Bit-Vector Levenshtein with DYNAMIC Partitioning (GPU-only) ===\n");

    // Load queries
    int numQueries = 0;
    char** querySeqs = loadQueries(&numQueries);
    if (!querySeqs || numQueries <= 0) {
        fprintf(stderr, "ERROR: Failed to load queries from %s\n", QUERY_FILE);
        return EXIT_FAILURE;
    }

    // Load references
    int numOrigRefs = 0;
    int* origRefLens = NULL;
    char** origRefs = loadReferences(&numOrigRefs, &origRefLens);
    if (!origRefs || numOrigRefs <= 0) {
        fprintf(stderr, "ERROR: Failed to load references from %s\n", REFERENCE_FILE);
        return EXIT_FAILURE;
    }

    printf("\n==================== LOADING ====================\n");
    printf("Queries Loaded: %d\n", numQueries);
    printf("Original References Loaded: %d\n", numOrigRefs);

    // ========================================================================
    // DYNAMIC PARTITIONING: Compute optimal parameters based on workload
    // ========================================================================
    int chunk_size, partition_threshold;
    
    printf("\n==================== DYNAMIC PARTITIONING ====================\n");
    compute_partitioning_strategy(
        numQueries, numOrigRefs, origRefLens,
        &chunk_size, &partition_threshold);

    // Partition references with computed parameters
    int queryLength0 = (int)strlen(querySeqs[0]);
    int overlap = queryLength0 - 1;

    PartitionedRefs partRefs = partition_references(
        origRefs, origRefLens, numOrigRefs,
        overlap, chunk_size, partition_threshold);

    printf("\n==================== PARTITIONING RESULTS ====================\n");
    printf("Overlap (Q-1): %d\n", overlap);
    printf("Generated Chunks: %d\n", partRefs.num_chunks);
    printf("Actual GPU blocks: %d queries × %d chunks = %d\n",
           numQueries, partRefs.num_chunks, numQueries * partRefs.num_chunks);

    // Compute query lengths
    int* queryLengths = computeQueryLengths(querySeqs, numQueries);

    // Build chunk mappings
    int* origChunkCounts;
    int** origChunkLists;
    build_orig_to_chunk_mapping(&partRefs, numOrigRefs,
                                &origChunkCounts, &origChunkLists);

    // Run algorithm
    runHyyroAlgorithm(numQueries, partRefs.num_chunks, numOrigRefs,
                     querySeqs, origRefs, origRefLens, &partRefs,
                     origChunkCounts, origChunkLists, queryLengths);

    // Cleanup
    free_orig_to_chunk_mapping(origChunkCounts, origChunkLists, numOrigRefs);
    free_partitioned_refs(&partRefs);
    
    for (int i = 0; i < numQueries; ++i) free(querySeqs[i]);
    free(querySeqs);
    
    for (int i = 0; i < numOrigRefs; ++i) free(origRefs[i]);
    free(origRefs);
    
    free(origRefLens);
    free(queryLengths);

    printf("\n==================== DONE ====================\n");

    return EXIT_SUCCESS;
}