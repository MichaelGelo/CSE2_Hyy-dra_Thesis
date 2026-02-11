// levenshtein_kernel.cuh
// CUDA kernel with CORRECT Hyyro-Damerau implementation
// FIXED: Last score now correctly tracks only the true final position

#ifndef LEVENSHTEIN_KERNEL_CUH
#define LEVENSHTEIN_KERNEL_CUH

#include <cuda_runtime.h>
#include <limits.h>
#include "bitvector.h"
#include "config.h"

// Debug control: Uncomment to enable score printing after each character
// #define DEBUG_PRINT_SCORES

// ============================================================================
// CORRECTED bitVectorDamerau - Matches pseudocode exactly
// Returns: final score, and also outputs the score at a specific position
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
    int targetPos,
    int* scoreAtTarget
)
{
    if (queryLength > 64 * BV_WORDS || queryLength <= 0) return -1;

    // Line 2: Bit-vector Pv, Mv, Ph, Mh, Xv, Xh, Eq, Xp
    bv_t Pv, Mv, Xv, Xh, Ph, Mh, Xp;

    // Temporary variables for operations
    bv_t tmp, tmp2, addtmp;
    bv_t Xh_or_Pv, not_Xh_or_Pv;
    bv_t Xh_or_Xv, not_Xh_or_Xv;
    bv_t Mh_shl, not_Xh, Xv_and_Pv;
    
    // Create mask and MSB info (computed once per call)
    bv_t queryMask;
    bvCreateMask(&queryMask, queryLength);
    int topWordIdx = (queryLength - 1) / 64;
    uint64_t topBitMask = 1ULL << ((queryLength - 1) % 64);

    // Line 3: Score = m
    int score = queryLength;

    // Line 4: Pv = 1^m
    bvSetAll(&Pv, ~0ULL);
    bvAnd(&Pv, &Pv, &queryMask);

    // Line 5: Mv = 0^m
    bvClear(&Mv);
    bvClear(&Xp);
    bvClear(&Xh);

    *zeroCount = 0; 
    *lowestCount = 0; 
    *lowestScore = score;
    if (scoreAtTarget) {
        *scoreAtTarget = -1;
    }

#ifdef DEBUG_PRINT_SCORES
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n*** INITIALIZATION ***\n");
        printf("queryLength=%d, queryMask.w[0]=0x%llx\n", queryLength, (unsigned long long)queryMask.w[0]);
        printf("AFTER Init: Pv=0x%llx, Mv=0x%llx, score=%d\n",
               (unsigned long long)Pv.w[0], (unsigned long long)Mv.w[0], score);
    }
#endif

    // Line 6: for j = 0, 1, 2, ..., n-1 do
    for (int j = 0; j < referenceLength; ++j) {
        unsigned char c = (unsigned char)reference[j];

        // Line 7: Eq = PEq[Σ[T[j]]]
        const bv_t* Eqc = &Eq[c];

#ifdef DEBUG_PRINT_SCORES
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\n=== Processing ref[%d] = '%c' ===\n", j, reference[j]);
            printf("Eq['%c'].w[0] = 0x%llx\n", c, (unsigned long long)Eqc->w[0]);
            printf("BEFORE: Pv=0x%llx, Mv=0x%llx, Xp=0x%llx, Xh=0x%llx\n",
                   (unsigned long long)Pv.w[0], (unsigned long long)Mv.w[0],
                   (unsigned long long)Xp.w[0], (unsigned long long)Xh.w[0]);
        }
#endif

        // Line 8: Xv = Eq | Mv
        bvOr(&Xv, Eqc, &Mv);

        // Line 9: Xh = (((~Xh) & Xv) << 1) & Xp
        bvNot(&not_Xh, &Xh);           // ~Xh
        bvAnd(&tmp, &not_Xh, &Xv);     // (~Xh) & Xv
        bvShl1(&tmp2, &tmp);           // ((~Xh) & Xv) << 1
        bvAnd(&Xh, &tmp2, &Xp);        // (((~Xh) & Xv) << 1) & Xp

        // Line 10: Xh = Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv
        bvAnd(&Xv_and_Pv, &Xv, &Pv);  // (Xv & Pv)
        bvAdd(&addtmp, &Xv_and_Pv, &Pv); // (Xv & Pv) + Pv
        bvXor(&tmp, &addtmp, &Pv);     // ((Xv & Pv) + Pv) ^ Pv
        bvOr(&Xh, &Xh, &tmp);          // Xh | (...)
        bvOr(&Xh, &Xh, &Xv);           // ... | Xv
        bvOr(&Xh, &Xh, &Mv);           // ... | Mv

        // Line 11: Ph = Mv | ~(Xh | Pv)
        bvOr(&Xh_or_Pv, &Xh, &Pv);
        bvNot(&not_Xh_or_Pv, &Xh_or_Pv);
        bvOr(&Ph, &Mv, &not_Xh_or_Pv);

        // Line 12: Mh = Xh & Pv
        bvAnd(&Mh, &Xh, &Pv);

        // Line 13: Xp = Xv
        bvCopy(&Xp, &Xv);

        // Line 14-15: Update score
        if (bvTestBitFast(&Ph, topWordIdx, topBitMask)) {
            ++score;
        } else if (bvTestBitFast(&Mh, topWordIdx, topBitMask)) {
            --score;
        }

#ifdef DEBUG_PRINT_SCORES
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Ph.w[0]=0x%llx, Mh.w[0]=0x%llx, Xh.w[0]=0x%llx, Xv.w[0]=0x%llx\n",
                   (unsigned long long)Ph.w[0], (unsigned long long)Mh.w[0],
                   (unsigned long long)Xh.w[0], (unsigned long long)Xv.w[0]);
            printf("Ph[%d]=%d, Mh[%d]=%d => score = %d\n",
                   queryLength-1, bvTestBitFast(&Ph, topWordIdx, topBitMask),
                   queryLength-1, bvTestBitFast(&Mh, topWordIdx, topBitMask), score);
        }
#endif

        // Track score at specific position (for last score tracking)
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

        // Line 16: Xv = (Ph << 1)
        bvShl1(&Xv, &Ph);

        // Line 17: Pv = (Mh << 1) | ~(Xh | Xv)
        // CRITICAL: Use the NEW Xv from line 16, not the old one!
        bvShl1(&Mh_shl, &Mh);          // Mh << 1
        bvOr(&Xh_or_Xv, &Xh, &Xv);     // Xh | Xv (using NEW Xv)
        bvNot(&not_Xh_or_Xv, &Xh_or_Xv); // ~(Xh | Xv)
        bvOr(&Pv, &Mh_shl, &not_Xh_or_Xv); // (Mh << 1) | ~(Xh | Xv)

        // Line 18: Mv = Xh & Xv
        bvAnd(&Mv, &Xh, &Xv);          // Use NEW Xv here too

        // Mask to pattern length
        bvAnd(&Pv, &Pv, &queryMask);
        bvAnd(&Mv, &Mv, &queryMask);

#ifdef DEBUG_PRINT_SCORES
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("AFTER: Pv=0x%llx, Mv=0x%llx (for next iteration)\n",
                   (unsigned long long)Pv.w[0], (unsigned long long)Mv.w[0]);
        }
#endif
    }

    return score;
}

// ============================================================================
// OPTIMIZED KERNEL WITH OFFSET-BASED MEMORY ACCESS
// FIXED: Only records last score when chunk actually ends at reference end
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

    // Load Eq table
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
            targetPos, &scoreAtTarget
        );

        d_pairDistances[pairIdx] = dist;
        d_pairZcounts[pairIdx] = localZeroCount;
        long long baseZptr = pairIdx * MAX_HITS;
        for (int k = 0; k < localZeroCount && k < MAX_HITS; ++k) {
            d_pairZindices[baseZptr + k] = chunkStart + localZeroIndices[k];
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
// SECOND PASS KERNEL WITH OFFSET-BASED ACCESS
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
            -1, NULL  // Don't track specific position
        );

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
