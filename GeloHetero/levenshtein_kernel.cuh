// levenshtein_kernel.cuh
// COMPLETELY FIXED Hyyro-Damerau implementation for heterogeneous system
// This version fixes BOTH the algorithm bugs AND the last score tracking

#ifndef LEVENSHTEIN_KERNEL_CUH
#define LEVENSHTEIN_KERNEL_CUH

#include <cuda_runtime.h>
#include <limits.h>
#include "bitvector.h"
#include "config.h"

// ============================================================================
// COMPLETELY CORRECTED bitVectorDamerau
// Fixed: Line 9 initialization, Lines 16-18 variable ordering
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
    int* lowestCount,
    int targetPos = -1,
    int* scoreAtTarget = NULL)
{
    if (queryLength > 64 * BV_WORDS || queryLength <= 0) return -1;

    // === Declare ALL bit-vectors ===
    bv_t Pv, Mv, Xv, Xh, Ph, Mh, Xp;
    bv_t tmp, tmp2, addtmp;
    bv_t Xh_or_Pv, not_Xh_or_Pv;
    bv_t Xh_or_Xv, not_Xh_or_Xv;
    bv_t Mh_shl, not_Xh, Xv_and_Pv;

    // Line 3: Score = m
    int score = queryLength;
    
    // Line 4: Pv = 1^m
    bvSetAll(&Pv, ~0ULL);
    bvMaskTop(&Pv, queryLength);
    
    // Line 5: Mv = 0^m
    bvClear(&Mv);
    
    // Initialize Xp and Xh to 0
    bvClear(&Xp);
    bvClear(&Xh);

    *zeroCount = 0;
    *lowestCount = 0;
    *lowestScore = score;
    if (scoreAtTarget) *scoreAtTarget = -1;

    // Line 6: for j = 1, 2, ..., n do
    for (int j = 0; j < referenceLength; ++j) {
        unsigned char c = (unsigned char)reference[j];
        const bv_t* Eqc = &Eq[c];

        // ==================== CRITICAL FIX #1 ====================
        // Line 8: Xv = Eq | Mv
        bvOr(&Xv, Eqc, &Mv);
        
        // Line 9: Xh = (((~Xh) & Xv) << 1) & Xp
        // THIS LINE WAS COMPLETELY MISSING IN YOUR CODE!
        bvNot(&not_Xh, &Xh);           // Step 1: ~Xh
        bvAnd(&tmp, &not_Xh, &Xv);     // Step 2: (~Xh) & Xv
        bvShl1(&tmp2, &tmp);           // Step 3: ((...) << 1)
        bvAnd(&Xh, &tmp2, &Xp);        // Step 4: (...) & Xp → Xh
        
        // Line 10: Xh = Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv
        bvAnd(&Xv_and_Pv, &Xv, &Pv);  // (Xv & Pv)
        bvAdd(&addtmp, &Xv_and_Pv, &Pv); // (Xv & Pv) + Pv
        bvXor(&tmp, &addtmp, &Pv);     // ((Xv & Pv) + Pv) ^ Pv
        bvOr(&Xh, &Xh, &tmp);          // Xh = Xh | (...)
        bvOr(&Xh, &Xh, &Xv);           // Xh = Xh | Xv
        bvOr(&Xh, &Xh, &Mv);           // Xh = Xh | Mv
        
        // Line 11: Ph = Mv | ~(Xh | Pv)
        bvOr(&Xh_or_Pv, &Xh, &Pv);
        bvNot(&not_Xh_or_Pv, &Xh_or_Pv);
        bvOr(&Ph, &Mv, &not_Xh_or_Pv);
        
        // Line 12: Mh = Xh & Pv
        bvAnd(&Mh, &Xh, &Pv);
        
        // Line 13: Xp = Xv (store OLD Xv before updating)
        bvCopy(&Xp, &Xv);

        // Line 14-15: Update score
        if (bvTestTop(&Ph, queryLength)) {
            ++score;
        } else if (bvTestTop(&Mh, queryLength)) {
            --score;
        }

        // Track score at target position
        if (scoreAtTarget && j == targetPos) {
            *scoreAtTarget = score;
        }

        // Track zero hits
        if (score == 0) {
            if (*zeroCount < MAX_HITS) {
                zeroIndices[*zeroCount] = j;
                (*zeroCount)++;
            }
        }

        // Track lowest score
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

        // ==================== CRITICAL FIX #2 ====================
        // Line 16: Xv = (Ph << 1)
        // MUST happen BEFORE lines 17-18 use the NEW Xv!
        bvShl1(&Xv, &Ph);
        
        // Line 17: Pv = (Mh << 1) | ~(Xh | Xv)
        // Uses the NEW Xv from line 16 (not the old one from line 8)!
        bvShl1(&Mh_shl, &Mh);
        bvOr(&Xh_or_Xv, &Xh, &Xv);     // Use NEW Xv here!
        bvNot(&not_Xh_or_Xv, &Xh_or_Xv);
        bvOr(&Pv, &Mh_shl, &not_Xh_or_Xv);
        
        // Line 18: Mv = Xh & Xv
        // Also uses the NEW Xv from line 16!
        bvAnd(&Mv, &Xh, &Xv);          // Use NEW Xv here!
        
        // Mask to pattern length
        bvMaskTop(&Pv, queryLength);
        bvMaskTop(&Mv, queryLength);
    }

    return score;
}

// ============================================================================
// KERNEL: FIRST PASS - Compute distances and find global minimum
// ============================================================================
__global__ 
void levenshteinKernelOptimized(
    int numQueries, int numChunks, int numOrigRefs,
    const char* __restrict__ d_queries,
    const int* __restrict__ d_qLens,
    const bv_t* __restrict__ d_EqQueries,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_refOffsets,
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
    
    int globalIdx = blockIdx.x;
    int totalPairs = numQueries * numChunks;
    
    if (globalIdx >= totalPairs) return;
    
    int q = globalIdx / numChunks;
    int c = globalIdx % numChunks;
    int tid = threadIdx.x;

    // Load Eq table into shared memory
    for (int i = tid; i < 256; i += blockDim.x) {
        sharedEq[i] = d_EqQueries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int queryLength = d_qLens[q];
    
    if (tid == 0) {
        const char* refptr = &d_refs[d_refOffsets[c]];
        int rlen = d_refLens[c];
        int chunkStart = d_chunkStarts[c];
        int orig = d_chunkToOrig[c];
        int origRefLen = d_origRefLens[orig];
        long long pairIdx = (long long)q * numChunks + c;

        int localZeroIndices[MAX_HITS];
        int localZeroCount = 0;
        int localLowestScore = INT_MAX;
        int localLowestIndices[MAX_HITS];
        int localLowestCount = 0;

        // Check if this chunk contains the true final position
        int chunkEnd = chunkStart + rlen;
        int targetPos = -1;
        int scoreAtTarget = -1;
        
        if (chunkEnd >= origRefLen) {
            targetPos = origRefLen - chunkStart - 1;
        }

        int dist = bitVectorDamerau(
            queryLength, refptr, rlen, sharedEq,
            localZeroIndices, &localZeroCount, 
            &localLowestScore, localLowestIndices, &localLowestCount,
            targetPos, &scoreAtTarget);

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

        long long origPairIdx = (long long)q * numOrigRefs + orig;
        if (localLowestCount > 0) {
            atomicMin(&d_lowestScoreOrig[origPairIdx], localLowestScore);
        }

        if (scoreAtTarget >= 0) {
            d_lastScoreOrig[origPairIdx] = scoreAtTarget;
        }
    }
    __syncthreads();
}

// ============================================================================
// KERNEL: SECOND PASS - Collect all positions with lowest score
// ============================================================================
__global__ 
void collectLowestIndicesKernel(
    int numQueries, int numChunks, int numOrigRefs,
    const char* __restrict__ d_queries,
    const int* __restrict__ d_qLens,
    const bv_t* __restrict__ d_EqQueries,
    const char* __restrict__ d_refs,
    const int* __restrict__ d_refLens,
    const int* __restrict__ d_refOffsets,
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

    for (int i = tid; i < 256; i += blockDim.x) {
        sharedEq[i] = d_EqQueries[(long long)q * 256LL + i];
    }
    __syncthreads();

    if (tid == 0) {
        const char* refptr = &d_refs[d_refOffsets[c]];
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
            &localLowestScore, localLowestIndices, &localLowestCount,
            -1, NULL);

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

#endif // LEVENSHTEIN_KERNEL_CUH