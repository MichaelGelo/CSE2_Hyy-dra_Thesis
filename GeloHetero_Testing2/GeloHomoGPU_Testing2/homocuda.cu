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
// COLLECT AND RETURN RESULTS FOR EACH PAIR
// ============================================================================
typedef struct {
    PairResult* results;
    int count;
} AllResults;

static AllResults collectAllResults(
    int numQueries, int numOrigRefs,
    int* origRefLens, 
    int* queryLengths,
    int* hostLowestScoreOrig, int* hostLowestCountOrig, int* hostLowestIndicesOrig,
    int* hostLastScoreOrig,
    const char* query_filename, const char* ref_filename)
{
    AllResults all_results = {0};
    all_results.count = numQueries * numOrigRefs;
    all_results.results = (PairResult*)malloc(sizeof(PairResult) * all_results.count);
    
    if (!all_results.results) {
        fprintf(stderr, "ERROR: Out of memory for results\n");
        all_results.count = 0;
        return all_results;
    }
    
    int result_idx = 0;
    
    for (int orig = 0; orig < numOrigRefs; ++orig) {
        for (int q = 0; q < numQueries; ++q) {
            // Get lowest score info directly from GPU results
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

            // Store result
            PairResult* res = &all_results.results[result_idx++];
            strncpy(res->query_filename, query_filename, sizeof(res->query_filename) - 1);
            res->query_filename[sizeof(res->query_filename) - 1] = '\0';
            res->query_length = queryLengths[q];
            
            strncpy(res->reference_filename, ref_filename, sizeof(res->reference_filename) - 1);
            res->reference_filename[sizeof(res->reference_filename) - 1] = '\0';
            res->reference_length = origRefLens[orig];
            
            res->num_hits = 0;  // Zero hits not collected in this simplified version
            res->hit_indexes = NULL;
            res->lowest_score = lowestScore;
            res->lowest_score_indexes = lowestIndices;
            res->lowest_score_count = (int)lowestCount;
            res->last_score = lastScore;

            // Print to console as well
            printf("Pair: Q(%d) Vs R(%d)\n", queryLengths[q], origRefLens[orig]);
            printf("Number of Hits: N/A\n");
            if (lowestScore != 0x7f7f7f7f) {
                printf("Lowest Score: %d\n", lowestScore);
                if (lowestCount > 0) {
                    printf("Lowest Score Indexes: ");
                    printIntArray(lowestIndices, lowestCount);
                    printf("\n");
                }
            }
            if (lastScore != 0x7f7f7f7f) {
                printf("Last Score: %d\n", lastScore);
            }
            printf("\n");
        }
    }
    
    return all_results;
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
int main() {
    printf("=== Hyyrö Bit-Vector Levenshtein with DYNAMIC Partitioning (GPU-only) ===\n");
    printf("=== MULTI-FILE MODE: Processing all FASTA files from folders ===\n\n");

    // Create results folder
    if (!create_results_folder(RESULTS_FOLDER)) {
        fprintf(stderr, "ERROR: Failed to create results folder\n");
        return EXIT_FAILURE;
    }

    // Scan for query and reference files
    printf("Scanning for FASTA files...\n");
    FastaFileList queryFiles = scan_folder_for_fasta(QUERY_FOLDER);
    FastaFileList refFiles = scan_folder_for_fasta(REFERENCE_FOLDER);
    
    printf("Found %d query file(s) and %d reference file(s)\n\n", queryFiles.count, refFiles.count);
    
    if (queryFiles.count == 0 || refFiles.count == 0) {
        fprintf(stderr, "ERROR: No FASTA files found in specified folders\n");
        return EXIT_FAILURE;
    }

    // Open CSV file for writing
    FILE* csv_file = fopen(RESULTS_CSV_FILE, "w");
    if (!csv_file) {
        fprintf(stderr, "ERROR: Failed to open CSV file: %s\n", RESULTS_CSV_FILE);
        return EXIT_FAILURE;
    }
    
    write_csv_header(csv_file);
    printf("CSV file created: %s\n\n", RESULTS_CSV_FILE);

    // Process each query file against all reference files
    for (int qi = 0; qi < queryFiles.count; ++qi) {
        printf("\n================== PROCESSING QUERY FILE %d/%d ==================\n", qi + 1, queryFiles.count);
        printf("Query: %s\n", queryFiles.filenames[qi]);
        
        // Load query file
        int numQueries = 0;
        char** querySeqs = parse_fasta_file(queryFiles.filenames[qi], &numQueries);
        if (!querySeqs || numQueries <= 0) {
            fprintf(stderr, "ERROR: Failed to load queries from %s\n", queryFiles.filenames[qi]);
            continue;
        }
        printf("Loaded %d queries\n\n", numQueries);

        // Process against each reference file
        for (int ri = 0; ri < refFiles.count; ++ri) {
            printf("\n---------- Processing against reference %d/%d ----------\n", ri + 1, refFiles.count);
            printf("Reference: %s\n", refFiles.filenames[ri]);
            
            // Load reference file
            int numOrigRefs = 0;
            int* origRefLens = NULL;
            char** origRefs = parse_fasta_file(refFiles.filenames[ri], &numOrigRefs);
            if (!origRefs || numOrigRefs <= 0) {
                fprintf(stderr, "ERROR: Failed to load references from %s\n", refFiles.filenames[ri]);
                continue;
            }
            printf("Loaded %d references\n", numOrigRefs);

            // Compute reference lengths
            origRefLens = (int*)malloc(sizeof(int) * numOrigRefs);
            if (!origRefLens) {
                fprintf(stderr, "ERROR: Out of memory for reference lengths\n");
                for (int i = 0; i < numOrigRefs; ++i) free(origRefs[i]);
                free(origRefs);
                continue;
            }
            
            for (int i = 0; i < numOrigRefs; ++i) {
                origRefLens[i] = (int)strlen(origRefs[i]);
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

            // Run algorithm and collect results
            printf("\n==================== RUNNING ALGORITHM ====================\n");
            
            // Build Eq tables
            bv_t* hostEqTables = buildEqTables(querySeqs, queryLengths, numQueries);
            if (!hostEqTables) {
                fprintf(stderr, "ERROR: Failed to build Eq tables\n");
                free_orig_to_chunk_mapping(origChunkCounts, origChunkLists, numOrigRefs);
                free_partitioned_refs(&partRefs);
                for (int i = 0; i < numQueries; ++i) free(querySeqs[i]);
                for (int i = 0; i < numOrigRefs; ++i) free(origRefs[i]);
                free(origRefLens);
                free(queryLengths);
                continue;
            }

            // Pack sequences EFFICIENTLY
            char* hostQueries = NULL;
            char* hostRefs = NULL;
            int* hostRefOffsets = NULL;
            size_t totalRefBytes = 0;
            
            packSequencesEfficient(querySeqs, numQueries, 
                                  partRefs.chunk_seqs, partRefs.chunk_lens, partRefs.num_chunks,
                                  &hostQueries, &hostRefs, &hostRefOffsets, &totalRefBytes);
            
            if (!hostQueries || !hostRefs || !hostRefOffsets) {
                fprintf(stderr, "ERROR: Out of memory for host buffers\n");
                free(hostEqTables);
                free_orig_to_chunk_mapping(origChunkCounts, origChunkLists, numOrigRefs);
                free_partitioned_refs(&partRefs);
                for (int i = 0; i < numQueries; ++i) free(querySeqs[i]);
                for (int i = 0; i < numOrigRefs; ++i) free(origRefs[i]);
                free(origRefLens);
                free(queryLengths);
                continue;
            }

            // Allocate GPU memory
            GpuBuffers gpuBuffers;
            allocateGpuMemory(&gpuBuffers, numQueries, partRefs.num_chunks, numOrigRefs, totalRefBytes);

            // Transfer data to GPU
            transferToGpuEfficient(&gpuBuffers, hostEqTables, hostQueries, hostRefs,
                                  queryLengths, partRefs.chunk_lens, hostRefOffsets,
                                  &partRefs, origRefLens,
                                  numQueries, partRefs.num_chunks, numOrigRefs, totalRefBytes);

            // Launch optimized kernel
            int totalPairs = numQueries * partRefs.num_chunks;
            int threads = THREADS_PER_BLOCK;
            int blocks = totalPairs;
            size_t sharedBytes = 256 * sizeof(bv_t);

            printf("*** OPTIMIZED Kernel launch: %d blocks × %d threads, shared=%zu bytes\n",
                   blocks, threads, sharedBytes);
            printf("    Total pairs: %d queries × %d chunks = %d blocks\n",
                   numQueries, partRefs.num_chunks, totalPairs);
            printf("    GPU utilization: %d parallel computations\n", blocks);

            cudaEvent_t ev_start, ev_end;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_end));

            float total_ms = 0.0f;

            for (int it = 0; it < LOOP_ITERATIONS; ++it) {
                CUDA_CHECK(cudaEventRecord(ev_start));

                levenshteinKernelOptimized<<<blocks, threads, sharedBytes>>>(
                    numQueries, partRefs.num_chunks, numOrigRefs,
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

            collectLowestIndicesKernel<<<blocks, threads, sharedBytes>>>(
                numQueries, partRefs.num_chunks, numOrigRefs,
                gpuBuffers.d_queries, gpuBuffers.d_qLens, gpuBuffers.d_EqQueries,
                gpuBuffers.d_refs, gpuBuffers.d_refLens, gpuBuffers.d_refOffsets,
                gpuBuffers.d_chunkStarts, gpuBuffers.d_chunkToOrig,
                gpuBuffers.d_lowestScoreOrig, gpuBuffers.d_lowestCountOrig,
                gpuBuffers.d_lowestIndicesOrig
            );

            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_end);

            // Copy results back to host
            long long totalPairChunks = (long long)numQueries * partRefs.num_chunks;
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

            // Collect and write results to CSV
            printf("\n==================== COLLECTING RESULTS ====================\n");
            AllResults results = collectAllResults(
                numQueries, numOrigRefs, origRefLens, queryLengths,
                hostLowestScoreOrig, hostLowestCountOrig, hostLowestIndicesOrig,
                hostLastScoreOrig,
                queryFiles.filenames[qi], refFiles.filenames[ri]);

            // Write all results to CSV
            for (int i = 0; i < results.count; ++i) {
                write_pair_result_csv(csv_file, &results.results[i]);
                
                // Cleanup result data
                if (results.results[i].hit_indexes) free(results.results[i].hit_indexes);
                if (results.results[i].lowest_score_indexes) free(results.results[i].lowest_score_indexes);
            }
            if (results.results) free(results.results);

            printf("Results written to CSV\n");

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
            
            free_orig_to_chunk_mapping(origChunkCounts, origChunkLists, numOrigRefs);
            free_partitioned_refs(&partRefs);
            
            for (int i = 0; i < numOrigRefs; ++i) free(origRefs[i]);
            free(origRefs);
            free(origRefLens);
            free(queryLengths);
        }

        // Cleanup query seqs
        for (int i = 0; i < numQueries; ++i) free(querySeqs[i]);
        free(querySeqs);
    }

    // Cleanup file lists
    free_fasta_file_list(&queryFiles);
    free_fasta_file_list(&refFiles);

    // Close CSV file
    fclose(csv_file);
    printf("\n==================== PROCESSING COMPLETE ====================\n");
    printf("Results saved to: %s\n", RESULTS_CSV_FILE);

    return EXIT_SUCCESS;
}