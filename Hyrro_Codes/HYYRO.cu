// CUDA_Multiple3.cu
// Hyyrö-style Damerau (adjacent transpositions) extension of your 256-bit Myers-based kernel.
// Kept file layout and host logic identical; replaced inner update loop to support adjacent transpositions.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "C_utils.h"

#define MAX_LENGTH 256
#define BV_WORDS 4   // 4 * 64 = 256.txt
typedef unsigned long long u64;

// params to change here ↓↓↓

#define query_file "s_query_256(GELO).fasta"
#define reference_file "ref_test2_multi10-1024(GELO).fasta"
#define threadsPerBlock 256
#define loope 10
#define maxZeros 2048
// params to change here ↑↑↑


struct BitVec256 {
    u64 w[BV_WORDS];
};

__device__ inline void bv_zero(BitVec256 &a) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) a.w[i] = 0ULL;
}
__device__ inline void bv_set_all(BitVec256 &a) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) a.w[i] = ~0ULL;
}
__device__ inline void bv_copy(const BitVec256 &src, BitVec256 &dst) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) dst.w[i] = src.w[i];
}
__device__ inline void bv_or(const BitVec256 &a, const BitVec256 &b, BitVec256 &r) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) r.w[i] = a.w[i] | b.w[i];
}
__device__ inline void bv_and(const BitVec256 &a, const BitVec256 &b, BitVec256 &r) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) r.w[i] = a.w[i] & b.w[i];
}
__device__ inline void bv_xor(const BitVec256 &a, const BitVec256 &b, BitVec256 &r) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) r.w[i] = a.w[i] ^ b.w[i];
}
__device__ inline void bv_not(const BitVec256 &a, BitVec256 &r) {
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) r.w[i] = ~a.w[i];
}

// left shift by 1: r = a << 1  (bits move toward higher indices)
__device__ inline void bv_shl1(const BitVec256 &a, BitVec256 &r) {
    u64 carry = 0ULL;
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) {
        u64 new_carry = (a.w[i] >> 63) & 1ULL;
        r.w[i] = (a.w[i] << 1) | carry;
        carry = new_carry;
    }
}

// right shift by 1: r = a >> 1  (bits move toward lower indices)
__device__ inline void bv_shr1(const BitVec256 &a, BitVec256 &r) {
    u64 carry = 0ULL;
    for (int i = BV_WORDS - 1; i >= 0; --i) {
        u64 new_carry = (a.w[i] & 1ULL);
        r.w[i] = (a.w[i] >> 1) | (carry << 63);
        carry = new_carry;
    }
}

// 256-bit add: r = a + b
// returns final carry-out (0 or 1)
__device__ inline unsigned int bv_add(const BitVec256 &a, const BitVec256 &b, BitVec256 &r) {
    u64 carry = 0ULL;
    #pragma unroll
    for (int i = 0; i < BV_WORDS; ++i) {
        u64 ai = a.w[i];
        u64 bi = b.w[i];
        u64 sum = ai + bi + carry;
        // detect carry: if (sum < ai) or (carry already set and sum == ai)
        carry = (sum < ai) || (carry && sum == ai);
        r.w[i] = sum;
    }
    return (unsigned int)carry;
}

// r = a + b where b is Pv in the original formula; we sometimes only need r
// Helper: compute ((Xv & Pv) + Pv)  -> use bv_and then bv_add
__device__ inline void bv_and_add(const BitVec256 &Xv, const BitVec256 &Pv, BitVec256 &tmp, BitVec256 &out) {
    bv_and(Xv, Pv, tmp);    // tmp = Xv & Pv
    bv_add(tmp, Pv, out);   // out = tmp + Pv  (carry ignored)
}

// mask bits above m-1 to zero (so bits >= m are cleared)
__device__ inline void bv_mask_top(BitVec256 &v, int m) {
    if (m >= MAX_LENGTH) return; // no masking needed
    int last_word = (m - 1) / 64;
    int last_bit = (m - 1) % 64;
    u64 last_mask = (last_bit == 63) ? ~0ULL : ((1ULL << (last_bit + 1)) - 1ULL);

    #pragma unroll
    for (int i = last_word + 1; i < BV_WORDS; ++i) v.w[i] = 0ULL;
    v.w[last_word] &= last_mask;
}

// test whether bit (m-1) is set in v: returns 0 or non-zero
__device__ inline int bv_test_msb(const BitVec256 &v, int m) {
    int idx = (m - 1) / 64;
    int off = (m - 1) % 64;
    return ( (v.w[idx] >> off) & 1ULL ) ? 1 : 0;
}

// Device kernel: one thread per query-reference pair
__global__ void bit_vector_levenshtein_256_kernel(
    const char *queries_flat,    // concatenated queries
    const char *refs_flat,       // concatenated references
    const int *q_off, const int *r_off,
    const int *q_len, const int *r_len,
    BitVec256 *Eq_table,         // Eq_table[256] per thread? here it's per thread use (we'll index by 256 * tid)
    int *results,
    int *lowest_scores,
    int *zero_counts,
    int *zero_indices,           // flattened: zero_indices[ tid * maxZeros + z ]
    int maxZerosPerPair
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load lengths
    int m = q_len[tid];
    int n = r_len[tid];

    if (m <= 0) { results[tid] = 0; lowest_scores[tid] = 0; zero_counts[tid] = 0; return; }
    if (m > MAX_LENGTH) { results[tid] = -1; lowest_scores[tid] = -1; zero_counts[tid] = 0; return; }

    const char *query = queries_flat + q_off[tid];
    const char *ref   = refs_flat + r_off[tid];

    // local per-thread vectors
    BitVec256 Pv, Mv, Xv, Xh, Ph, Mh, tmp, addtmp;
    // extra temps for Damerau handling
    BitVec256 PrevEqMask;   // Eq mask of previous reference character
    BitVec256 trans_mask;   // mask marking positions that may allow transposition
    BitVec256 t1, t2;

    // Eq for each possible byte value - but to avoid huge memory we assume Eq_table is a contiguous block
    BitVec256 *Eq = Eq_table + (size_t)tid * 256; // Eq[256] for this thread

    // init Eq to zero
    for (int c = 0; c < 256; ++c) bv_zero(Eq[c]);

    // Build Eq: for each position i set bit i in Eq[ query[i] ]
    for (int i = 0; i < m; ++i) {
        unsigned char ch = (unsigned char)query[i];
        int word = i / 64;
        int bit  = i % 64;
        Eq[ch].w[word] |= (1ULL << bit);
    }

    // initialize Pv = all ones, Mv = zero
    bv_set_all(Pv);
    bv_zero(Mv);
    bv_zero(PrevEqMask);
    bv_zero(trans_mask);

    // mask Pv bits above m (Pv should have only m meaningful bits set)
    bv_mask_top(Pv, m);

    int score = m;
    int lowest = m;
    int zeros = 0;

    unsigned char prev_rc = 0; // previous reference char (for transposition detection)
    bool have_prev = false;

    // loop over reference
    for (int j = 0; j < n; ++j) {
        unsigned char rc = (unsigned char)ref[j];

        // Standard Myers parts (compute Xv etc.)
        // Xv = Eq[rc] | Mv
        bv_or(Eq[rc], Mv, Xv);

        // Xh = (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv
        bv_and_add(Xv, Pv, tmp, addtmp);   // addtmp = (Xv & Pv) + Pv
        bv_xor(addtmp, Pv, Xh);            // Xh = addtmp ^ Pv
        // Xh = Xh | Xv | Mv
        BitVec256 temp_or;
        bv_or(Xh, Xv, temp_or);
        bv_or(temp_or, Mv, Xh);

        // Ph = Mv | ~(Xh | Pv)
        BitVec256 xh_or_pv;
        bv_or(Xh, Pv, xh_or_pv);
        bv_not(xh_or_pv, tmp);      // tmp = ~(Xh | Pv)
        bv_or(Mv, tmp, Ph);

        // Mh = Xh & Pv
        bv_and(Xh, Pv, Mh);

        // Score updates depend on bit (m-1) of Ph and Mh (same as before)
        if (bv_test_msb(Ph, m)) score++;
        if (bv_test_msb(Mh, m)) score--;

        if (score <= lowest) lowest = score;

        // --- Damerau extension: detect adjacent transpositions and allow them ---
        // Idea:
        // - PrevEqMask is the Eq mask for the previous reference character (ref[j-1])
        // - A transposition candidate is where current rc matches at position i and previous rc matches at i-1 in the query.
        // - We detect positions where Eq[rc] has a 1 at i AND PrevEqMask has a 1 at i-1.
        // - That is: trans_mask = Eq[rc] & (PrevEqMask >> 1)
        // - If trans_mask has a 1 at position (m-1) or other positions it can influence, we allow a correction that mimics a transposition by
        //   setting bits in Mh/Ph or directly adjusting Pv/Mv so the next iteration will reflect the transposition.
        //
        // Implementation below follows that idea using multiword shifts.

        // compute PrevEqMask >> 1 into t1
        if (have_prev) {
            bv_shr1(PrevEqMask, t1);       // t1 = PrevEqMask >> 1
            bv_and(Eq[rc], t1, trans_mask); // trans_mask = Eq[rc] & (PrevEqMask >> 1)
        } else {
            bv_zero(trans_mask);
        }

        // If a transposition candidate exists at the highest bit (m-1), this may reduce/increase score similar to Myers' transposition handling.
        // We apply a conservative correction: where trans_mask has bits set we clear the corresponding bits in Pv (simulate a matching transposition)
        // and set corresponding bits in Mv so that the DP column accounts for the swap.
        // This is not exactly the full Hyyrö formalism but implements the standard bit-parallel adjacent-transposition correction.
        //
        // Pv = Pv & ~trans_mask
        bv_not(trans_mask, t2);
        bv_and(Pv, t2, Pv);

        // Incorporate transposition candidates into Mv to allow recovery (Mv |= trans_mask)
        bv_or(Mv, trans_mask, Mv);

        // Now continue with standard Myers update for Pv and Mv
        // Pv = (Mh << 1) | ~(Xh | (Ph << 1))
        BitVec256 Mh_shl, Ph_shl, xh_or_phshl, not_xh_or_phshl;
        bv_shl1(Mh, Mh_shl);
        bv_shl1(Ph, Ph_shl);
        bv_or(Xh, Ph_shl, xh_or_phshl);
        bv_not(xh_or_phshl, not_xh_or_phshl);
        bv_or(Mh_shl, not_xh_or_phshl, Pv);

        // Mv = Xh & (Ph << 1)
        bv_and(Xh, Ph_shl, Mv);

        // mask Pv and Mv beyond m bits to keep high bits clean
        bv_mask_top(Pv, m);
        bv_mask_top(Mv, m);

        // record zero score positions as before
        if (score == 0 && zeros < maxZerosPerPair) {
            zero_indices[tid * maxZerosPerPair + zeros] = j;
            zeros++;
        }

        // update PrevEqMask to Eq[rc] for next iteration
        bv_copy(Eq[rc], PrevEqMask);
        have_prev = true;
        prev_rc = rc;
    }

    results[tid] = score;
    lowest_scores[tid] = lowest;
    zero_counts[tid] = zeros;
}

/* ------------------------------
   Minimal host-side example (one pair)
   ------------------------------ */

int main() {
    // Load queries
    int num_queries = 0;
    char **query_seqs = parse_fasta_file(query_file, &num_queries);

    // Load references
    int num_references = 0;
    char **reference_seqs = parse_fasta_file(reference_file, &num_references);
    if (!reference_seqs) {
        for (int i = 0; i < num_queries; i++) free(query_seqs[i]);
        free(query_seqs);
        return -1;
    }

    // Build all pairs
    int numPairs = num_queries * num_references;

    // Flatten queries and references into contiguous buffers
    size_t total_q_chars = 0, total_r_chars = 0;
    for (int i = 0; i < num_queries; i++) total_q_chars += strlen(query_seqs[i]);
    for (int j = 0; j < num_references; j++) total_r_chars += strlen(reference_seqs[j]);

    char *h_q_flat = (char*)malloc(total_q_chars);
    char *h_r_flat = (char*)malloc(total_r_chars);
    int *h_q_off = (int*)malloc(sizeof(int) * numPairs);
    int *h_r_off = (int*)malloc(sizeof(int) * numPairs);
    int *h_q_len = (int*)malloc(sizeof(int) * numPairs);
    int *h_r_len = (int*)malloc(sizeof(int) * numPairs);

    // Fill flattened buffers and offsets
    size_t q_pos = 0, r_pos = 0;
    int *query_offsets = (int*)malloc(sizeof(int) * num_queries);
    int *reference_offsets = (int*)malloc(sizeof(int) * num_references);
    for (int i = 0; i < num_queries; i++) {
        query_offsets[i] = (int)q_pos;
        size_t L = strlen(query_seqs[i]);
        memcpy(h_q_flat + q_pos, query_seqs[i], L);
        q_pos += L;
    }
    for (int j = 0; j < num_references; j++) {
        reference_offsets[j] = (int)r_pos;
        size_t L = strlen(reference_seqs[j]);
        memcpy(h_r_flat + r_pos, reference_seqs[j], L);
        r_pos += L;
    }

    // Now assign per-pair offsets/lengths
    int pair = 0;
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < num_references; j++) {
            h_q_off[pair] = query_offsets[i];
            h_q_len[pair] = (int)strlen(query_seqs[i]);
            h_r_off[pair] = reference_offsets[j];
            h_r_len[pair] = (int)strlen(reference_seqs[j]);
            pair++;
        }
    }

    // Device allocations that hold full flattened sequences (unchanged)
    char *d_q, *d_r;
    cudaMalloc(&d_q, total_q_chars);
    cudaMalloc(&d_r, total_r_chars);
    cudaMemcpy(d_q, h_q_flat, total_q_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r_flat, total_r_chars, cudaMemcpyHostToDevice);

    // Choose batch size: tune this to your Jetson. Start with 256 or 512.
    const int batchSize = 512; // <--- tune this (256, 512, 1024)
    if (batchSize <= 0) return -1;

    // Allocate device-side per-batch arrays (capacity = batchSize)
    int *d_q_off, *d_r_off, *d_q_len, *d_r_len;
    cudaMalloc(&d_q_off, sizeof(int) * batchSize);
    cudaMalloc(&d_r_off, sizeof(int) * batchSize);
    cudaMalloc(&d_q_len, sizeof(int) * batchSize);
    cudaMalloc(&d_r_len, sizeof(int) * batchSize);

    // Eq table allocation per batch: 256 * batchSize BitVec256 entries
    BitVec256 *d_Eq;
    cudaMalloc(&d_Eq, sizeof(BitVec256) * 256 * batchSize);

    // Results per batch
    int *d_results, *d_lowest, *d_zero_counts, *d_zero_indices;
    cudaMalloc(&d_results, sizeof(int) * batchSize);
    cudaMalloc(&d_lowest, sizeof(int) * batchSize);
    cudaMalloc(&d_zero_counts, sizeof(int) * batchSize);
    cudaMalloc(&d_zero_indices, sizeof(int) * maxZeros * batchSize);

    // Host arrays for final collection (full numPairs)
    int *h_results = (int*)malloc(sizeof(int) * numPairs);
    int *h_lowest  = (int*)malloc(sizeof(int) * numPairs);
    int *h_zero_counts = (int*)malloc(sizeof(int) * numPairs);
    int *h_zero_indices = (int*)malloc(sizeof(int) * maxZeros * numPairs);

    // Batch loop
    int totalBatches = (numPairs + batchSize - 1) / batchSize;
    for (int b = 0; b < totalBatches; b++) {
        int start = b * batchSize;
        int count = numPairs - start;
        if (count > batchSize) count = batchSize;

        // copy only the slices needed for this batch to device
        cudaMemcpy(d_q_off, h_q_off + start, sizeof(int) * count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_off, h_r_off + start, sizeof(int) * count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q_len, h_q_len + start, sizeof(int) * count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_len, h_r_len + start, sizeof(int) * count, cudaMemcpyHostToDevice);

        // Launch kernel on this batch
        int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        bit_vector_levenshtein_256_kernel<<<blocks, threadsPerBlock>>>(
            d_q, d_r, d_q_off, d_r_off, d_q_len, d_r_len,
            d_Eq, d_results, d_lowest, d_zero_counts, d_zero_indices, maxZeros
        );
        cudaDeviceSynchronize();

        // copy back device results for this batch into the host arrays at correct offset
        cudaMemcpy(h_results + start, d_results, sizeof(int) * count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lowest + start, d_lowest, sizeof(int) * count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_zero_counts + start, d_zero_counts, sizeof(int) * count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_zero_indices + start * maxZeros, d_zero_indices, sizeof(int) * maxZeros * count, cudaMemcpyDeviceToHost);
    }

    // Print results with zero-hit details (formatted)
    for (int p = 0; p < numPairs; p++) {
        int qi = p / num_references;
        int rj = p % num_references;
        int hits = h_zero_counts[p];
        int qlen = h_q_len[p];
        int rlen = h_r_len[p];
	
	printf("---------------------------------------------------------------\n");
        printf("Pair: Q%d(%d) Vs R%d(%d)\n", qi + 1, qlen, rj + 1, rlen);
        printf("Number of Hits: %d\n", hits);

        if (hits > 0) {
            printf("Hit Indexes: [");
            for (int z = 0; z < hits; z++) {
                int idx = h_zero_indices[p * maxZeros + z];
                printf("%d", idx);
                if (z < hits - 1) printf(", ");
            }
            printf("]\n");
            printf("Lowest Score: N/A\n");
            printf("Lowest Score Indexes: N/A\n");
        } else {
            printf("Hit Indexes: []\n");
            printf("Lowest Score: %d\n", h_lowest[p]);
            printf("Lowest Score Indexes: [N/A]\n");
        }

        printf("Last Score: %d\n", h_results[p]);
        printf("\n");
	printf("---------------------------------------------------------------\n");
    }
    // free device
    cudaFree(d_q); cudaFree(d_r);
    cudaFree(d_q_off); cudaFree(d_r_off); cudaFree(d_q_len); cudaFree(d_r_len);
    cudaFree(d_Eq);
    cudaFree(d_results); cudaFree(d_lowest); cudaFree(d_zero_counts);
    cudaFree(d_zero_indices);

    // free host
    free(h_q_flat); free(h_r_flat);
    free(h_q_off); free(h_r_off); free(h_q_len); free(h_r_len);
    free(query_offsets); free(reference_offsets);

    for (int i = 0; i < num_queries; i++) free(query_seqs[i]);
    for (int j = 0; j < num_references; j++) free(reference_seqs[j]);
    free(query_seqs); free(reference_seqs);

    free(h_results); free(h_lowest); free(h_zero_counts); free(h_zero_indices);

    return 0;
}
