#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <cuda_runtime.h>
#include "C_utils.h"

// --- WINDOWS DLL EXPORT DEFINITION ---
#ifdef _WIN32
#define ALIGNMENT_API __declspec(dllexport)
#else
#define ALIGNMENT_API
#endif
// --------------------------------------

// --- CONFIGURATION ---
#define MAX_LENGTH (1 << 26)
#define MAX_HITS 1024
#define threadsPerBlock 256
#define CHUNK_SIZE 166700
#define PARTITION_THRESHOLD 1000000
#define BV_WORDS 4

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define CUDA_CHECK(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1;                                                                                 \
        }                                                                                              \
    } while (0)

// --- BV helpers (Copied from original) ---
typedef struct
{
    uint64_t w[BV_WORDS];
} bv_t;

static __forceinline__ __host__ __device__ void bv_set_all_unrolled(bv_t *out, uint64_t v)
{
    out->w[0] = v;
    out->w[1] = v;
    out->w[2] = v;
    out->w[3] = v;
}
static __forceinline__ __host__ __device__ void bv_clear_unrolled(bv_t *out)
{
    out->w[0] = 0ULL;
    out->w[1] = 0ULL;
    out->w[2] = 0ULL;
    out->w[3] = 0ULL;
}
static __forceinline__ __host__ __device__ void bv_copy_unrolled(bv_t *out, const bv_t *in)
{
    out->w[0] = in->w[0];
    out->w[1] = in->w[1];
    out->w[2] = in->w[2];
    out->w[3] = in->w[3];
}
static __forceinline__ __host__ __device__ void bv_or_unrolled(bv_t *out, const bv_t *a, const bv_t *b)
{
    out->w[0] = a->w[0] | b->w[0];
    out->w[1] = a->w[1] | b->w[1];
    out->w[2] = a->w[2] | b->w[2];
    out->w[3] = a->w[3] | b->w[3];
}
static __forceinline__ __host__ __device__ void bv_and_unrolled(bv_t *out, const bv_t *a, const bv_t *b)
{
    out->w[0] = a->w[0] & b->w[0];
    out->w[1] = a->w[1] & b->w[1];
    out->w[2] = a->w[2] & b->w[2];
    out->w[3] = a->w[3] & b->w[3];
}
static __forceinline__ __host__ __device__ void bv_not_unrolled(bv_t *out, const bv_t *a)
{
    out->w[0] = ~(a->w[0]);
    out->w[1] = ~(a->w[1]);
    out->w[2] = ~(a->w[2]);
    out->w[3] = ~(a->w[3]);
}
static __forceinline__ __host__ __device__ void bv_shl1_unrolled(bv_t *out, const bv_t *in)
{
    uint64_t c0 = in->w[0] >> 63;
    uint64_t c1 = in->w[1] >> 63;
    uint64_t c2 = in->w[2] >> 63;
    out->w[0] = (in->w[0] << 1);
    out->w[1] = (in->w[1] << 1) | c0;
    out->w[2] = (in->w[2] << 1) | c1;
    out->w[3] = (in->w[3] << 1) | c2;
}
static __forceinline__ __host__ __device__ int bv_test_top_unrolled(const bv_t *v, int query_length)
{
    int idx = (query_length - 1) / 64;
    int bit = (query_length - 1) % 64;
    if (idx == 0)
        return ((v->w[0] >> bit) & 1ULL) ? 1 : 0;
    if (idx == 1)
        return ((v->w[1] >> bit) & 1ULL) ? 1 : 0;
    if (idx == 2)
        return ((v->w[2] >> bit) & 1ULL) ? 1 : 0;
    return ((v->w[3] >> bit) & 1ULL) ? 1 : 0;
}
static __forceinline__ __host__ __device__ uint64_t bv_add_unrolled(bv_t *out, const bv_t *a, const bv_t *b)
{
    uint64_t carry = 0ULL;
    uint64_t sum0 = a->w[0] + b->w[0];
    carry = (sum0 < a->w[0]);
    out->w[0] = sum0;
    uint64_t sum1 = a->w[1] + b->w[1] + carry;
    carry = (sum1 < a->w[1]) || (sum1 == a->w[1] && carry);
    out->w[1] = sum1;
    uint64_t sum2 = a->w[2] + b->w[2] + carry;
    carry = (sum2 < a->w[2]) || (sum2 == a->w[2] && carry);
    out->w[2] = sum2;
    uint64_t sum3 = a->w[3] + b->w[3] + carry;
    carry = (sum3 < a->w[3]) || (sum3 == a->w[3] && carry);
    out->w[3] = sum3;
    return carry;
}
static __forceinline__ __host__ __device__ void bv_xor_unrolled(bv_t *out, const bv_t *a, const bv_t *b)
{
    out->w[0] = a->w[0] ^ b->w[0];
    out->w[1] = a->w[1] ^ b->w[1];
    out->w[2] = a->w[2] ^ b->w[2];
    out->w[3] = a->w[3] ^ b->w[3];
}

// --- Bit-vector Levenshtein (Copied from original) ---
static __forceinline__ __host__ __device__ int bit_vector_levenshtein_local(
    int query_length,
    const char *reference,
    int reference_length,
    const bv_t *Eq,
    int *zero_indices,
    int *zero_count,
    int *lowest,
    int *lowest_index_local)
{
    if (query_length > 64 * BV_WORDS || query_length <= 0)
        return -1;

    bv_t Pv, Mv, Ph, Mh, Xv, Xh, Xp;
    bv_set_all_unrolled(&Pv, ~0ULL);
    bv_clear_unrolled(&Mv);
    bv_clear_unrolled(&Ph);
    bv_clear_unrolled(&Mh);
    bv_clear_unrolled(&Xv);
    bv_clear_unrolled(&Xh);
    bv_clear_unrolled(&Xp);

    *zero_count = 0;
    int score = query_length;
    *lowest = score;
    *lowest_index_local = -1;

    bv_t tmp1, tmp2, tmp3, tmp4;

    for (int j = 0; j < reference_length; ++j)
    {
        unsigned char c = (unsigned char)reference[j];
        const bv_t *Eqc = &Eq[c];

        bv_or_unrolled(&Xv, Eqc, &Mv);

        bv_not_unrolled(&tmp1, &Xh);
        bv_and_unrolled(&tmp2, &tmp1, &Xv);
        bv_shl1_unrolled(&tmp3, &tmp2);
        bv_and_unrolled(&tmp4, &tmp3, &Xp);
        bv_copy_unrolled(&Xh, &tmp4);

        bv_and_unrolled(&tmp1, &Xv, &Pv);
        bv_add_unrolled(&tmp2, &tmp1, &Pv);
        bv_xor_unrolled(&tmp2, &tmp2, &Pv);
        bv_or_unrolled(&tmp3, &tmp2, &Xv);

        bv_or_unrolled(&tmp2, &tmp3, &Mv);
        bv_or_unrolled(&tmp1, &Xh, &tmp2);
        bv_copy_unrolled(&Xh, &tmp1);

        bv_or_unrolled(&tmp1, &Xh, &Pv);
        bv_not_unrolled(&tmp2, &tmp1);
        bv_or_unrolled(&Ph, &Mv, &tmp2);

        bv_and_unrolled(&Mh, &Xh, &Pv);

        if (bv_test_top_unrolled(&Ph, query_length))
            ++score;
        if (bv_test_top_unrolled(&Mh, query_length))
            --score;

        if (score < *lowest)
        {
            *lowest = score;
            *lowest_index_local = j;
        }

        bv_copy_unrolled(&Xp, &Xv);

        bv_shl1_unrolled(&Xv, &Ph);

        bv_shl1_unrolled(&tmp1, &Mh);
        bv_or_unrolled(&tmp2, &Xh, &Xv);
        bv_not_unrolled(&tmp3, &tmp2);
        bv_or_unrolled(&Pv, &tmp1, &tmp3);

        bv_and_unrolled(&Mv, &Xh, &Xv);

        if (score == 0)
        {
            if (*zero_count < MAX_HITS)
                zero_indices[*zero_count] = j;
            (*zero_count)++;
        }
    }

    return score;
}

// --- Kernel (Copied from original) ---
__global__ void levenshtein_kernel_shared_agg(
    int num_queries, int num_chunks, int num_orig_refs,
    const char *__restrict__ d_queries, const int *__restrict__ d_q_lens, const bv_t *__restrict__ d_Eq_queries,
    const char *__restrict__ d_refs, const int *__restrict__ d_ref_lens, // per-chunk
    const int *__restrict__ d_chunk_starts, const int *__restrict__ d_chunk_to_orig,
    const int *__restrict__ d_orig_ref_lens, // per original ref
    // pair-level outputs (per query x chunk)
    int *__restrict__ d_pair_distances, int *__restrict__ d_pair_zcounts, int *__restrict__ d_pair_zindices,
    // per-original aggregated (per query x original ref)
    int *__restrict__ d_lowest_score_orig, int *__restrict__ d_lowest_index_orig, int *__restrict__ d_last_score_orig)
{
    extern __shared__ bv_t s_Eq[]; // 256 entries
    int q = blockIdx.x;
    if (q >= num_queries)
        return;
    int tid = threadIdx.x;

    // load Eq for this query into shared
    for (int i = tid; i < 256; i += blockDim.x)
    {
        s_Eq[i] = d_Eq_queries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int qlen = d_q_lens[q];
    for (int c = tid; c < num_chunks; c += blockDim.x)
    {
        const char *refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_ref_lens[c];
        int chunk_start = d_chunk_starts[c];
        int orig = d_chunk_to_orig[c]; // original reference id
        long long pair_idx = (long long)q * num_chunks + c;

        // local arrays (stack)
        int local_zero_indices[MAX_HITS];
        int local_zero_count = 0;
        int local_lowest_val = INT_MAX;
        int local_lowest_idx = -1;

        int dist = bit_vector_levenshtein_local(qlen, refptr, rlen, s_Eq,
                                                local_zero_indices, &local_zero_count, &local_lowest_val, &local_lowest_idx);

        // write pair-level results (pair-level arrays sized num_queries * num_chunks)
        d_pair_distances[pair_idx] = dist;

        // -------- GPU-side aggregation for original reference (per query x orig) --------
        long long orig_pair_idx = (long long)q * num_orig_refs + orig;

        // 1) update lowest score and lowest index atomically (atomic CAS loop)
        if (local_lowest_idx >= 0)
        {
            int global_lowest_pos = chunk_start + local_lowest_idx;
            int old;
            do
            {
                old = d_lowest_score_orig[orig_pair_idx];
                if (local_lowest_val < old)
                { // attempt to replace
                    int res = atomicCAS(&d_lowest_score_orig[orig_pair_idx], old, local_lowest_val);
                    if (res == old)
                    {
                        atomicExch(&d_lowest_index_orig[orig_pair_idx], global_lowest_pos);
                        break;
                    }
                }
                else
                {
                    break; // no update needed
                }
            } while (true);
        }
    }
}

// =========================================================================
// === C-CALLABLE PYTHON INTERFACE FUNCTION (NOW EXPORTED) =================
// =========================================================================
extern "C" ALIGNMENT_API int align_sequences_gpu(
    const char *h_query,
    int q_len,
    const char *h_reference,
    int r_len,
    int *out_lowest_score,
    int *out_lowest_index,
    int *out_last_score)
{
    // --- 0. VALIDATION AND SETUP ---
    if (q_len <= 0 || r_len <= 0 || q_len > 64 * BV_WORDS)
        return -1;

    // Simplification: Handle only one query and one reference, and assume no chunking.
    int num_orig_refs = 1;
    int num_queries = 1;
    int num_chunks = 1;

    // --- 1. BUILD HOST BUFFERS AND Eq ---
    // ... (Host buffer and Eq initialization logic remains the same)

    // a) Build h_Eq_queries (Only for query 0)
    bv_t *h_Eq_queries = (bv_t *)malloc(256 * sizeof(bv_t));
    if (!h_Eq_queries)
        return -2;
    memset(h_Eq_queries, 0, 256 * sizeof(bv_t));
    for (int i = 0; i < q_len; ++i)
    {
        unsigned char ch = (unsigned char)h_query[i];
        int word = i / 64;
        int bit = i % 64;
        h_Eq_queries[ch].w[word] |= (1ULL << bit);
    }

    // b) Build contiguous host buffers
    char *h_queries = (char *)malloc(MAX_LENGTH);
    char *h_refs = (char *)malloc(MAX_LENGTH);
    if (!h_queries || !h_refs)
    {
        free(h_Eq_queries);
        return -3;
    }
    // Use strncpy but be mindful of null termination; q_len and r_len are sufficient
    memcpy(h_queries, h_query, q_len);
    memcpy(h_refs, h_reference, r_len);

    // c) Small arrays for the kernel (size 1)
    int h_ref_lens[1] = {r_len};
    int h_chunk_starts[1] = {0};
    int h_chunk_to_orig[1] = {0};
    int h_orig_ref_lens[1] = {r_len};
    int h_q_lens[1] = {q_len};

    // --- 2. DEVICE ALLOCATIONS AND COPY ---
    bv_t *d_Eq_queries = NULL;
    char *d_queries = NULL;
    char *d_refs = NULL;
    int *d_q_lens = NULL;
    int *d_ref_lens = NULL;
    int *d_chunk_starts = NULL;
    int *d_chunk_to_orig = NULL;
    int *d_orig_ref_lens = NULL;
    int *d_pair_distances = NULL;
    int *d_lowest_score_orig = NULL;
    int *d_lowest_index_orig = NULL;

    int ret_val = 0; // Use ret_val to manage cleanup on failure

    // Allocate Device Memory
    if (cudaMalloc((void **)&d_Eq_queries, 256 * sizeof(bv_t)) != cudaSuccess)
    {
        ret_val = -4;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_queries, q_len) != cudaSuccess)
    {
        ret_val = -5;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_refs, r_len) != cudaSuccess)
    {
        ret_val = -6;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_q_lens, sizeof(int)) != cudaSuccess)
    {
        ret_val = -7;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_ref_lens, sizeof(int)) != cudaSuccess)
    {
        ret_val = -8;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_chunk_starts, sizeof(int)) != cudaSuccess)
    {
        ret_val = -9;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_chunk_to_orig, sizeof(int)) != cudaSuccess)
    {
        ret_val = -10;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_orig_ref_lens, sizeof(int)) != cudaSuccess)
    {
        ret_val = -11;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_pair_distances, num_chunks * sizeof(int)) != cudaSuccess)
    {
        ret_val = -12;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_lowest_score_orig, num_orig_refs * sizeof(int)) != cudaSuccess)
    {
        ret_val = -13;
        goto cleanup;
    }
    if (cudaMalloc((void **)&d_lowest_index_orig, num_orig_refs * sizeof(int)) != cudaSuccess)
    {
        ret_val = -14;
        goto cleanup;
    }

    // Copy Host to Device
    if (cudaMemcpy(d_Eq_queries, h_Eq_queries, 256 * sizeof(bv_t), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -15;
        goto cleanup;
    }
    if (cudaMemcpy(d_queries, h_queries, q_len, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -16;
        goto cleanup;
    }
    if (cudaMemcpy(d_refs, h_refs, r_len, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -17;
        goto cleanup;
    }
    if (cudaMemcpy(d_q_lens, h_q_lens, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -18;
        goto cleanup;
    }
    if (cudaMemcpy(d_ref_lens, h_ref_lens, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -19;
        goto cleanup;
    }
    if (cudaMemcpy(d_chunk_starts, h_chunk_starts, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -20;
        goto cleanup;
    }
    if (cudaMemcpy(d_chunk_to_orig, h_chunk_to_orig, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -21;
        goto cleanup;
    }
    if (cudaMemcpy(d_orig_ref_lens, h_orig_ref_lens, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        ret_val = -22;
        goto cleanup;
    }

    // Initialize lowest score sentinel (INT_MAX) and index (-1)
    if (cudaMemset(d_lowest_score_orig, 0x7f, num_orig_refs * sizeof(int)) != cudaSuccess)
    {
        ret_val = -23;
        goto cleanup;
    }
    if (cudaMemset(d_lowest_index_orig, 0xff, num_orig_refs * sizeof(int)) != cudaSuccess)
    {
        ret_val = -24;
        goto cleanup;
    }

    // --- 3. LAUNCH KERNEL (loope=1) ---
    size_t shared_bytes = 256 * sizeof(bv_t);
    levenshtein_kernel_shared_agg<<<num_queries, threadsPerBlock, shared_bytes>>>(
        num_queries, num_chunks, num_orig_refs,
        d_queries, d_q_lens, d_Eq_queries,
        d_refs, d_ref_lens,
        d_chunk_starts, d_chunk_to_orig,
        d_orig_ref_lens,
        d_pair_distances, /* d_pair_zcounts */ NULL, /* d_pair_zindices */ NULL, // Omit large Z-hit arrays
        d_lowest_score_orig, d_lowest_index_orig, /* d_last_score_orig */ NULL);
    if (cudaGetLastError() != cudaSuccess)
    {
        ret_val = -25;
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        ret_val = -26;
        goto cleanup;
    }

    // --- 4. RETRIEVE RESULTS ---
    int h_lowest_score = 0;
    int h_lowest_index = -1;
    int h_last_score = 0;

    // Retrieve aggregated lowest score/index for the original ref (index 0)
    if (cudaMemcpy(&h_lowest_score, d_lowest_score_orig, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        ret_val = -27;
        goto cleanup;
    }
    if (cudaMemcpy(&h_lowest_index, d_lowest_index_orig, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        ret_val = -28;
        goto cleanup;
    }

    // Retrieve last score from the single chunk's pair distance (index 0)
    if (cudaMemcpy(&h_last_score, d_pair_distances, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        ret_val = -29;
        goto cleanup;
    }

    // Write back to Python output pointers
    *out_lowest_score = h_lowest_score;
    *out_lowest_index = h_lowest_index;
    *out_last_score = h_last_score;

    // --- 5. CLEANUP ---
cleanup:
    if (d_Eq_queries)
        cudaFree(d_Eq_queries);
    if (d_queries)
        cudaFree(d_queries);
    if (d_refs)
        cudaFree(d_refs);
    if (d_q_lens)
        cudaFree(d_q_lens);
    if (d_ref_lens)
        cudaFree(d_ref_lens);
    if (d_chunk_starts)
        cudaFree(d_chunk_starts);
    if (d_chunk_to_orig)
        cudaFree(d_chunk_to_orig);
    if (d_orig_ref_lens)
        cudaFree(d_orig_ref_lens);
    if (d_pair_distances)
        cudaFree(d_pair_distances);
    if (d_lowest_score_orig)
        cudaFree(d_lowest_score_orig);
    if (d_lowest_index_orig)
        cudaFree(d_lowest_index_orig);

    free(h_Eq_queries);
    free(h_queries);
    free(h_refs);

    return ret_val; // 0 for success, negative for error
}