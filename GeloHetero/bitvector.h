// bitvector.h
// Bit-vector operations and Eq table construction for Levenshtein computation

#ifndef BITVECTOR_H
#define BITVECTOR_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// CONSTANTS
// ============================================================================
#define BV_WORDS 4

// ============================================================================
// BIT-VECTOR TYPE
// 256-bit vector using 4x 64-bit words for pattern matching
// ============================================================================
typedef struct { 
    uint64_t w[BV_WORDS]; 
} bv_t;

// ============================================================================
// BIT-VECTOR OPERATIONS
// Optimized operations for both host and device (GPU) execution
// ============================================================================

static __forceinline__ __host__ __device__ 
void bvSetAll(bv_t* out, uint64_t v) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = v;
    }
}

static __forceinline__ __host__ __device__ 
void bvClear(bv_t* out) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = 0ULL;
    }
}

static __forceinline__ __host__ __device__ 
void bvCopy(bv_t* out, const bv_t* in) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = in->w[i];
    }
}

static __forceinline__ __host__ __device__ 
void bvOr(bv_t* out, const bv_t* a, const bv_t* b) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = a->w[i] | b->w[i];
    }
}

static __forceinline__ __host__ __device__ 
void bvAnd(bv_t* out, const bv_t* a, const bv_t* b) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = a->w[i] & b->w[i];
    }
}

static __forceinline__ __host__ __device__ 
void bvXor(bv_t* out, const bv_t* a, const bv_t* b) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = a->w[i] ^ b->w[i];
    }
}

static __forceinline__ __host__ __device__ 
void bvNot(bv_t* out, const bv_t* a) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = ~(a->w[i]);
    }
}

static __forceinline__ __host__ __device__ 
void bvShl1(bv_t* out, const bv_t* in) {
    uint64_t carry[BV_WORDS - 1];
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS - 1; ++i) {
        carry[i] = in->w[i] >> 63;
    }
    
    out->w[0] = (in->w[0] << 1);
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 1; i < BV_WORDS; ++i) {
        out->w[i] = (in->w[i] << 1) | carry[i - 1];
    }
}

static __forceinline__ __host__ __device__ 
void bvShr1(bv_t* out, const bv_t* in) {
    uint64_t carry[BV_WORDS - 1];
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = BV_WORDS - 1; i > 0; --i) {
        carry[i - 1] = in->w[i] & 1ULL;
    }
    
    out->w[BV_WORDS - 1] = (in->w[BV_WORDS - 1] >> 1);
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = BV_WORDS - 2; i >= 0; --i) {
        out->w[i] = (in->w[i] >> 1) | (carry[i] << 63);
    }
}

static __forceinline__ __host__ __device__ 
int bvTestTop(const bv_t* v, int queryLength) {
    int idx = (queryLength - 1) / 64;
    int bit = (queryLength - 1) % 64;
    return ((v->w[idx] >> bit) & 1ULL) ? 1 : 0;
}

static __forceinline__ __host__ __device__
uint64_t bvAdd(bv_t* out, const bv_t* a, const bv_t* b) {
    uint64_t carry = 0ULL;

    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        uint64_t sum = a->w[i] + b->w[i] + carry;
        carry = (sum < a->w[i]) || (sum == a->w[i] && carry);
        out->w[i] = sum;
    }

    return carry;
}

static __forceinline__ __host__ __device__ 
void bvMaskTop(bv_t *v, int m) {
    if (m >= 64 * BV_WORDS) return;
    
    int lastWord = (m - 1) / 64;
    int lastBit = (m - 1) % 64;
    uint64_t lastMask = (lastBit == 63) ? ~0ULL : ((1ULL << (lastBit + 1)) - 1ULL);
    
    v->w[lastWord] &= lastMask;
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = lastWord + 1; i < BV_WORDS; ++i) {
        v->w[i] = 0ULL;
    }
}

// ============================================================================
// EQ TABLE CONSTRUCTION
// Builds character equality lookup tables for pattern matching
// ============================================================================

static inline bv_t* buildEqTables(char** queries, int* queryLengths, int numQueries) {
    bv_t* eqTables = (bv_t*)malloc((size_t)numQueries * 256 * sizeof(bv_t));
    if (!eqTables) {
        fprintf(stderr, "ERROR: Out of memory for Eq tables\n");
        return NULL;
    }
    
    memset(eqTables, 0, (size_t)numQueries * 256 * sizeof(bv_t));
    
    for (int q = 0; q < numQueries; ++q) {
        int qlen = queryLengths[q];
        const char* queryStr = queries[q];
        
        for (int i = 0; i < qlen; ++i) {
            unsigned char ch = (unsigned char)queryStr[i];
            int word = i / 64;
            int bit = i % 64;
            eqTables[(long long)q * 256 + ch].w[word] |= (1ULL << bit);
        }
    }
    
    return eqTables;
}

// Create a static mask where valid bits are 1 and invalid bits are 0
static __forceinline__ __host__ __device__
void bvCreateMask(bv_t* out, int m) {
    bvClear(out);
    if (m <= 0) return;

    int lastWord = (m - 1) / 64;
    int lastBit = (m - 1) % 64;

    // Set all full words to 1s
    for (int i = 0; i < lastWord; ++i) {
        out->w[i] = ~0ULL;
    }

    // Set the partial word
    uint64_t lastMask = (lastBit == 63) ? ~0ULL : ((1ULL << (lastBit + 1)) - 1ULL);
    out->w[lastWord] = lastMask;

    // Remaining words are already 0 from bvClear
}

// Fast check using pre-computed index and mask
static __forceinline__ __host__ __device__
int bvTestBitFast(const bv_t* v, int wordIdx, uint64_t bitMask) {
    return (v->w[wordIdx] & bitMask) ? 1 : 0;
}

#endif // BITVECTOR_H