// bitvector.h
// Bit-vector operations for Levenshtein distance computation
// Provides optimized 256-bit vector operations using 4x 64-bit words

#ifndef BITVECTOR_H
#define BITVECTOR_H

#include <stdint.h>

// ============================================================================
// CONSTANTS
// ============================================================================
#define BV_WORDS 4

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================
typedef struct { 
    uint64_t w[BV_WORDS]; 
} bv_t;

// ============================================================================
// BIT-VECTOR OPERATIONS
// Sets all words in bit-vector to the specified value
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

// ============================================================================
// Clears all bits in the bit-vector to zero
// ============================================================================
static __forceinline__ __host__ __device__ 
void bvClear(bv_t* out) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = 0ULL;
    }
}

// ============================================================================
// Copies bit-vector from source to destination
// ============================================================================
static __forceinline__ __host__ __device__ 
void bvCopy(bv_t* out, const bv_t* in) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = in->w[i];
    }
}

// ============================================================================
// Bitwise OR operation: out = a | b
// ============================================================================
static __forceinline__ __host__ __device__ 
void bvOr(bv_t* out, const bv_t* a, const bv_t* b) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = a->w[i] | b->w[i];
    }
}

// ============================================================================
// Bitwise AND operation: out = a & b
// ============================================================================
static __forceinline__ __host__ __device__ 
void bvAnd(bv_t* out, const bv_t* a, const bv_t* b) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = a->w[i] & b->w[i];
    }
}

// ============================================================================
// Bitwise XOR operation: out = a ^ b
// ============================================================================
static __forceinline__ __host__ __device__ 
void bvXor(bv_t* out, const bv_t* a, const bv_t* b) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = a->w[i] ^ b->w[i];
    }
}

// ============================================================================
// Bitwise NOT operation: out = ~a
// ============================================================================
static __forceinline__ __host__ __device__ 
void bvNot(bv_t* out, const bv_t* a) {
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for (int i = 0; i < BV_WORDS; ++i) {
        out->w[i] = ~(a->w[i]);
    }
}

// ============================================================================
// Left shift by 1 bit with carry propagation across words
// ============================================================================
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

// ============================================================================
// Right shift by 1 bit with carry propagation across words
// ============================================================================
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

// ============================================================================
// Tests if the bit at position (queryLength - 1) is set
// Used to determine if edit distance increases or decreases
// ============================================================================
static __forceinline__ __host__ __device__ 
int bvTestTop(const bv_t* v, int queryLength) {
    int idx = (queryLength - 1) / 64;
    int bit = (queryLength - 1) % 64;
    return ((v->w[idx] >> bit) & 1ULL) ? 1 : 0;
}

// ============================================================================
// Multi-precision addition: out = a + b, returns final carry
// ============================================================================
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

// ============================================================================
// Masks bits above position m-1 (clears bits >= m)
// Used to ensure bit-vectors don't exceed query length
// ============================================================================
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

#endif // BITVECTOR_H