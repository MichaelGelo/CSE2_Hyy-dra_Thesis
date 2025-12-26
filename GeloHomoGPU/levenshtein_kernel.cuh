// levenshtein_kernel.cuh
// CUDA kernel implementation for Hyyrö bit-vector Damerau-Levenshtein distance
// Includes device functions and kernel launch wrapper

#ifndef LEVENSHTEIN_KERNEL_CUH
#define LEVENSHTEIN_KERNEL_CUH

#include <cuda_runtime.h>
#include <limits.h>
#include "bitvector.h"
#include "config.h"

// ============================================================================
// CORE DAMERAU-LEVENSHTEIN COMPUTATION (DEVICE FUNCTION)
// Computes Damerau-Levenshtein distance using Hyyrö's bit-vector algorithm
// Handles insertions, deletions, substitutions, AND transpositions
// Tracks zero scores (exact matches) and lowest scores with their positions
// ============================================================================
static __forceinline__ __host__ __device__ 
int bitVectorDamerau(
    int queryLength,
    const char* reference,
    int referenceLength,
    const bv_t* Eq,
    int* zeroIndices,
    int* zeroCount,
    int* lowestScore,
    int* lowestIndices,
    int* lowestCount)
{
    if (queryLength > 64 * BV_WORDS || queryLength <= 0) return -1;

    // Bit vectors for Hyyrö-Damerau algorithm
    bv_t Pv, Mv, Xv, Xh, Ph, Mh, Xp;
    bv_t tmp, tmp2, addtmp, Xh_or_Pv, not_Xh_or_Pv, Xh_or_Xv, not_Xh_or_Xv;
    bv_t Mh_shl, Ph_shl, not_Xh, Xv_and_Pv;

    // Initialize vectors according to Hyyrö-Damerau pseudocode
    bvSetAll(&Pv, ~0ULL);  // Pv = 1^m
    bvClear(&Mv);          // Mv = 0^m
    bvClear(&Xp);          // Xp = 0 (stores previous Xv)
    bvClear(&Xh);          // Xh starts at 0
    bvMaskTop(&Pv, queryLength);

    *zeroCount = 0;
    *lowestCount = 0;
    int score = queryLength;
    *lowestScore = score;

    for (int j = 0; j < referenceLength; ++j) {
        unsigned char c = (unsigned char)reference[j];
        const bv_t* Eqc = &Eq[c];

        // Line 8: Xv = Eq | Mv
        bvOr(&Xv, Eqc, &Mv);
        
        // Line 9: Xh = (((~Xh) & Xv) << 1) & Xp
        // This is the transposition detection step
        bvNot(&not_Xh, &Xh);
        bvAnd(&tmp, &not_Xh, &Xv);
        bvShl1(&tmp2, &tmp);
        bvAnd(&Xh, &tmp2, &Xp);
        
        // Line 10: Xh = Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv
        bvAnd(&Xv_and_Pv, &Xv, &Pv);
        bvAdd(&addtmp, &Xv_and_Pv, &Pv);
        bvXor(&tmp, &addtmp, &Pv);
        bvOr(&Xh, &Xh, &tmp);
        bvOr(&Xh, &Xh, &Xv);
        bvOr(&Xh, &Xh, &Mv);
        
        // Line 11: Ph = Mv | ~(Xh | Pv)
        bvOr(&Xh_or_Pv, &Xh, &Pv);
        bvNot(&not_Xh_or_Pv, &Xh_or_Pv);
        bvOr(&Ph, &Mv, &not_Xh_or_Pv);
        
        // Line 12: Mh = Xh & Pv
        bvAnd(&Mh, &Xh, &Pv);
        
        // Line 13: Xp = Xv (store current pattern bit-vector for next iteration)
        bvCopy(&Xp, &Xv);
        
        // Line 14-15: Update score
        if (bvTestTop(&Ph, queryLength)) {
            ++score;
        } else if (bvTestTop(&Mh, queryLength)) {
            --score;
        }

        // Track lowest scores
        if (score < *lowestScore) {
            *lowestScore = score;
            *lowestCount = 0;
        }
        if (score == *lowestScore) {
            if (*lowestCount < MAX_HITS) {
                lowestIndices[*lowestCount] = j;
                (*lowestCount)++;
            }
        }

        // Line 16: Xv = (Ph << 1)
        bvShl1(&Xv, &Ph);
        
        // Line 17: Pv = (Mh << 1) | ~(Xh | Xv)
        bvShl1(&Mh_shl, &Mh);
        bvOr(&Xh_or_Xv, &Xh, &Xv);
        bvNot(&not_Xh_or_Xv, &Xh_or_Xv);
        bvOr(&Pv, &Mh_shl, &not_Xh_or_Xv);
        
        // Line 18: Mv = Xh & Xv
        bvAnd(&Mv, &Xh, &Xv);
        
        // Mask to pattern length
        bvMaskTop(&Pv, queryLength);
        bvMaskTop(&Mv, queryLength);

        // Track zero scores (exact matches)
        if (score == 0) {
            if (*zeroCount < MAX_HITS) {
                zeroIndices[*zeroCount] = j;
                (*zeroCount)++;
            }
        }
    }

    return score;
}

// ============================================================================
// PHASE 1: COMPUTE DISTANCES AND UPDATE LOWEST SCORES
// Each thread processes multiple chunks, computing distances and tracking
// the minimum score across all chunks for each original reference
// ============================================================================
__device__ 
void phase1ComputeDistances(
    int q, int tid, int numQueries, int numChunks, int numOrigRefs,
    int queryLength,
    const bv_t* sharedEq,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_chunkStarts,
    const int* __restrict__ d_chunkToOrig,
    const int* __restrict__ d_origRefLens,
    int* __restrict__ d_pairDistances,
    int* __restrict__ d_pairZcounts,
    int* __restrict__ d_pairZindices,
    int* __restrict__ d_lowestScoreOrig,
    int* __restrict__ d_lastScoreOrig)
{
    for (int c = tid; c < numChunks; c += blockDim.x) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_refLens[c];
        int chunkStart = d_chunkStarts[c];
        int orig = d_chunkToOrig[c];
        long long pairIdx = (long long)q * numChunks + c;

        int localZeroIndices[MAX_HITS];
        int localZeroCount = 0;
        int localLowestScore = INT_MAX;
        int localLowestIndices[MAX_HITS];
        int localLowestCount = 0;

        int dist = bitVectorDamerau(
            queryLength, refptr, rlen, sharedEq,
            localZeroIndices, &localZeroCount, 
            &localLowestScore, localLowestIndices, &localLowestCount);

        // Store pair-level results
        d_pairDistances[pairIdx] = dist;
        d_pairZcounts[pairIdx] = localZeroCount;
        long long baseZptr = pairIdx * MAX_HITS;
        for (int k = 0; k < localZeroCount && k < MAX_HITS; ++k) {
            int globalPos = chunkStart + localZeroIndices[k];
            d_pairZindices[baseZptr + k] = globalPos;
        }
        for (int k = localZeroCount; k < MAX_HITS; ++k) {
            d_pairZindices[baseZptr + k] = -1;
        }

        // Update lowest score for original reference
        long long origPairIdx = (long long)q * numOrigRefs + orig;
        if (localLowestCount > 0) {
            atomicMin(&d_lowestScoreOrig[origPairIdx], localLowestScore);
        }

        // Track last score for this original reference
        if (chunkStart + rlen == d_origRefLens[orig]) {
            d_lastScoreOrig[origPairIdx] = dist;
        }
    }
}

// ============================================================================
// PHASE 2: COLLECT INDICES MATCHING FINAL BEST SCORES
// After all scores are computed, collect position indices that match
// the globally best score for each original reference
// ============================================================================
__device__ 
void phase2CollectIndices(
    int q, int tid, int numQueries, int numChunks, int numOrigRefs,
    int queryLength,
    const bv_t* sharedEq,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_chunkStarts,
    const int* __restrict__ d_chunkToOrig,
    const int* __restrict__ d_lowestScoreOrig,
    int* __restrict__ d_lowestCountOrig,
    int* __restrict__ d_lowestIndicesOrig)
{
    for (int c = tid; c < numChunks; c += blockDim.x) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_refLens[c];
        int chunkStart = d_chunkStarts[c];
        int orig = d_chunkToOrig[c];

        int localZeroIndices[MAX_HITS];
        int localZeroCount = 0;
        int localLowestScore = INT_MAX;
        int localLowestIndices[MAX_HITS];
        int localLowestCount = 0;

        bitVectorDamerau(
            queryLength, refptr, rlen, sharedEq,
            localZeroIndices, &localZeroCount, 
            &localLowestScore, localLowestIndices, &localLowestCount);

        long long origPairIdx = (long long)q * numOrigRefs + orig;
        long long origIndicesBase = origPairIdx * MAX_HITS;
        int finalBestScore = d_lowestScoreOrig[origPairIdx];
        
        if (localLowestScore == finalBestScore && localLowestCount > 0) {
            for (int k = 0; k < localLowestCount; ++k) {
                int globalLowestPos = chunkStart + localLowestIndices[k];
                int slot = atomicAdd(&d_lowestCountOrig[origPairIdx], 1);
                if (slot < MAX_HITS) {
                    d_lowestIndicesOrig[origIndicesBase + slot] = globalLowestPos;
                }
            }
        }
    }
}

// ============================================================================
// OPTIMIZED KERNEL: ONE BLOCK PER QUERY-CHUNK PAIR
// Each block computes distance for one query-chunk pair
// Much better GPU utilization: blocks = num_queries × num_chunks
// ============================================================================
__global__ 
void levenshteinKernelOptimized(
    int numQueries, int numChunks, int numOrigRefs,
    const char* __restrict__ d_queries,
    const int* __restrict__ d_qLens,
    const bv_t* __restrict__ d_EqQueries,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_chunkStarts,
    const int* __restrict__ d_chunkToOrig,
    const int* __restrict__ d_origRefLens,
    int* __restrict__ d_pairDistances,
    int* __restrict__ d_pairZcounts,
    int* __restrict__ d_pairZindices,
    int* __restrict__ d_lowestScoreOrig,
    int* __restrict__ d_lowestCountOrig,
    int* __restrict__ d_lowestIndicesOrig,
    int* __restrict__ d_lastScoreOrig)
{
    extern __shared__ bv_t sharedEq[];
    
    // Each block handles one query-chunk pair
    int globalIdx = blockIdx.x;
    int totalPairs = numQueries * numChunks;
    
    if (globalIdx >= totalPairs) return;
    
    // Decode which query and chunk this block processes
    int q = globalIdx / numChunks;
    int c = globalIdx % numChunks;
    
    int tid = threadIdx.x;

    // Load Eq table into shared memory (cooperative load)
    for (int i = tid; i < 256; i += blockDim.x) {
        sharedEq[i] = d_EqQueries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int queryLength = d_qLens[q];
    
    // Only thread 0 computes (others helped load Eq table)
    if (tid == 0) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_refLens[c];
        int chunkStart = d_chunkStarts[c];
        int orig = d_chunkToOrig[c];
        long long pairIdx = (long long)q * numChunks + c;

        int localZeroIndices[MAX_HITS];
        int localZeroCount = 0;
        int localLowestScore = INT_MAX;
        int localLowestIndices[MAX_HITS];
        int localLowestCount = 0;

        int dist = bitVectorDamerau(
            queryLength, refptr, rlen, sharedEq,
            localZeroIndices, &localZeroCount, 
            &localLowestScore, localLowestIndices, &localLowestCount);

        // Store pair-level results
        d_pairDistances[pairIdx] = dist;
        d_pairZcounts[pairIdx] = localZeroCount;
        long long baseZptr = pairIdx * MAX_HITS;
        for (int k = 0; k < localZeroCount && k < MAX_HITS; ++k) {
            int globalPos = chunkStart + localZeroIndices[k];
            d_pairZindices[baseZptr + k] = globalPos;
        }
        for (int k = localZeroCount; k < MAX_HITS; ++k) {
            d_pairZindices[baseZptr + k] = -1;
        }

        // Update lowest score for original reference
        long long origPairIdx = (long long)q * numOrigRefs + orig;
        if (localLowestCount > 0) {
            atomicMin(&d_lowestScoreOrig[origPairIdx], localLowestScore);
        }

        // Track last score
        if (chunkStart + rlen == d_origRefLens[orig]) {
            d_lastScoreOrig[origPairIdx] = dist;
        }
    }
    __syncthreads();
}

// ============================================================================
// SECOND PASS KERNEL: COLLECT INDICES MATCHING BEST SCORES
// Runs after first kernel to gather all positions with minimum distance
// ============================================================================
__global__ 
void collectLowestIndicesKernel(
    int numQueries, int numChunks, int numOrigRefs,
    const char* __restrict__ d_queries,
    const int* __restrict__ d_qLens,
    const bv_t* __restrict__ d_EqQueries,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_chunkStarts,
    const int* __restrict__ d_chunkToOrig,
    const int* __restrict__ d_lowestScoreOrig,
    int* __restrict__ d_lowestCountOrig,
    int* __restrict__ d_lowestIndicesOrig)
{
    extern __shared__ bv_t sharedEq[];
    
    int globalIdx = blockIdx.x;
    int totalPairs = numQueries * numChunks;
    
    if (globalIdx >= totalPairs) return;
    
    int q = globalIdx / numChunks;
    int c = globalIdx % numChunks;
    int tid = threadIdx.x;

    // Load Eq table
    for (int i = tid; i < 256; i += blockDim.x) {
        sharedEq[i] = d_EqQueries[(long long)q * 256LL + i];
    }
    __syncthreads();

    if (tid == 0) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_refLens[c];
        int chunkStart = d_chunkStarts[c];
        int orig = d_chunkToOrig[c];
        int queryLength = d_qLens[q];

        int localZeroIndices[MAX_HITS];
        int localZeroCount = 0;
        int localLowestScore = INT_MAX;
        int localLowestIndices[MAX_HITS];
        int localLowestCount = 0;

        bitVectorDamerau(
            queryLength, refptr, rlen, sharedEq,
            localZeroIndices, &localZeroCount, 
            &localLowestScore, localLowestIndices, &localLowestCount);

        long long origPairIdx = (long long)q * numOrigRefs + orig;
        long long origIndicesBase = origPairIdx * MAX_HITS;
        int finalBestScore = d_lowestScoreOrig[origPairIdx];
        
        if (localLowestScore == finalBestScore && localLowestCount > 0) {
            for (int k = 0; k < localLowestCount; ++k) {
                int globalLowestPos = chunkStart + localLowestIndices[k];
                int slot = atomicAdd(&d_lowestCountOrig[origPairIdx], 1);
                if (slot < MAX_HITS) {
                    d_lowestIndicesOrig[origIndicesBase + slot] = globalLowestPos;
                }
            }
        }
    }
}

// ============================================================================
// LEGACY KERNEL (for backward compatibility)
// Original design: one block per query, threads iterate over chunks
// ============================================================================
__global__ 
void levenshteinKernel(
    int numQueries, int numChunks, int numOrigRefs,
    const char* __restrict__ d_queries,
    const int* __restrict__ d_qLens,
    const bv_t* __restrict__ d_EqQueries,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_chunkStarts,
    const int* __restrict__ d_chunkToOrig,
    const int* __restrict__ d_origRefLens,
    int* __restrict__ d_pairDistances,
    int* __restrict__ d_pairZcounts,
    int* __restrict__ d_pairZindices,
    int* __restrict__ d_lowestScoreOrig,
    int* __restrict__ d_lowestCountOrig,
    int* __restrict__ d_lowestIndicesOrig,
    int* __restrict__ d_lastScoreOrig)
{
    extern __shared__ bv_t sharedEq[];
    
    int q = blockIdx.x;
    if (q >= numQueries) return;
    
    int tid = threadIdx.x;

    // Load Eq table into shared memory
    for (int i = tid; i < 256; i += blockDim.x) {
        sharedEq[i] = d_EqQueries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int queryLength = d_qLens[q];
    
    // Phase 1: Compute distances and update lowest scores
    phase1ComputeDistances(
        q, tid, numQueries, numChunks, numOrigRefs,
        queryLength, sharedEq,
        d_refs, d_refLens, d_chunkStarts, d_chunkToOrig, d_origRefLens,
        d_pairDistances, d_pairZcounts, d_pairZindices,
        d_lowestScoreOrig, d_lastScoreOrig);
    
    // Synchronize before Phase 2
    __syncthreads();
    __threadfence();
    
    // Phase 2: Collect indices matching final best scores
    phase2CollectIndices(
        q, tid, numQueries, numChunks, numOrigRefs,
        queryLength, sharedEq,
        d_refs, d_refLens, d_chunkStarts, d_chunkToOrig,
        d_lowestScoreOrig, d_lowestCountOrig, d_lowestIndicesOrig);
}

#endif // LEVENSHTEIN_KERNEL_CUH