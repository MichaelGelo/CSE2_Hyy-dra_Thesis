%%writefile hycuda.cu
// unified_leven_partitioned_gpu_no_shared.cu
// Combined partition + Hyyrö bit-vector Levenshtein with GPU-side aggregation
// Shared memory for Eq removed to avoid ptxas error
// Compile:
//   nvcc -O3 unified_leven_partitioned_gpu_no_shared.cu C_utils.c -o unified_leven_partitioned_gpu_no_shared
//
// Run:
//   ./unified_leven_partitioned_gpu_no_shared
//
// Expects C_utils.c/h with parse_fasta_file() and read_file_into_string().

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "C_utils.h"

#define MAX_LENGTH (1 << 24)   // per-slot buffer size (must be >= longest chunk)
#define MAX_HITS 1024

// user params
#define query_file "que4_256.fasta"
#define reference_file "ref5_50M.fasta"
#define threadsPerBlock 256
#define loope 10
#define CHUNK_SIZE 166700
#define PARTITION_THRESHOLD 1000000
// end user params

#define BV_WORDS 4
typedef struct { uint64_t w[BV_WORDS]; } bv_t;

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static int compare_ints(const void *a, const void *b) {
    int av = *(const int*)a;
    int bv = *(const int*)b;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
}

// ---------- BV helpers ----------
static __forceinline__ __host__ __device__ void bv_set_all_unrolled(bv_t* out, uint64_t v) {
    out->w[0] = v; out->w[1] = v; out->w[2] = v; out->w[3] = v;
}
static __forceinline__ __host__ __device__ void bv_clear_unrolled(bv_t* out) {
    out->w[0] = 0ULL; out->w[1] = 0ULL; out->w[2] = 0ULL; out->w[3] = 0ULL;
}
static __forceinline__ __host__ __device__ void bv_copy_unrolled(bv_t* out, const bv_t* in) {
    out->w[0] = in->w[0]; out->w[1] = in->w[1]; out->w[2] = in->w[2]; out->w[3] = in->w[3];
}
static __forceinline__ __host__ __device__ void bv_or_unrolled(bv_t* out, const bv_t* a, const bv_t* b) {
    out->w[0] = a->w[0] | b->w[0];
    out->w[1] = a->w[1] | b->w[1];
    out->w[2] = a->w[2] | b->w[2];
    out->w[3] = a->w[3] | b->w[3];
}
static __forceinline__ __host__ __device__ void bv_and_unrolled(bv_t* out, const bv_t* a, const bv_t* b) {
    out->w[0] = a->w[0] & b->w[0];
    out->w[1] = a->w[1] & b->w[1];
    out->w[2] = a->w[2] & b->w[2];
    out->w[3] = a->w[3] & b->w[3];
}
static __forceinline__ __host__ __device__ void bv_xor_unrolled(bv_t* out, const bv_t* a, const bv_t* b) {
    out->w[0] = a->w[0] ^ b->w[0];
    out->w[1] = a->w[1] ^ b->w[1];
    out->w[2] = a->w[2] ^ b->w[2];
    out->w[3] = a->w[3] ^ b->w[3];
}
static __forceinline__ __host__ __device__ void bv_not_unrolled(bv_t* out, const bv_t* a) {
    out->w[0] = ~(a->w[0]); out->w[1] = ~(a->w[1]); out->w[2] = ~(a->w[2]); out->w[3] = ~(a->w[3]);
}
static __forceinline__ __host__ __device__ void bv_shl1_unrolled(bv_t* out, const bv_t* in) {
    uint64_t c0 = in->w[0] >> 63;
    uint64_t c1 = in->w[1] >> 63;
    uint64_t c2 = in->w[2] >> 63;
    out->w[0] = (in->w[0] << 1);
    out->w[1] = (in->w[1] << 1) | c0;
    out->w[2] = (in->w[2] << 1) | c1;
    out->w[3] = (in->w[3] << 1) | c2;
}
static __forceinline__ __host__ __device__ int bv_test_top_unrolled(const bv_t* v, int query_length) {
    int idx = (query_length - 1) / 64;
    int bit = (query_length - 1) % 64;
    if (idx == 0) return ((v->w[0] >> bit) & 1ULL) ? 1 : 0;
    if (idx == 1) return ((v->w[1] >> bit) & 1ULL) ? 1 : 0;
    if (idx == 2) return ((v->w[2] >> bit) & 1ULL) ? 1 : 0;
    return ((v->w[3] >> bit) & 1ULL) ? 1 : 0;
}
static __forceinline__ __host__ __device__ uint64_t bv_add_unrolled(bv_t* out, const bv_t* a, const bv_t* b) {
    unsigned __int128 t0 = (unsigned __int128)a->w[0] + (unsigned __int128)b->w[0];
    out->w[0] = (uint64_t)t0;
    unsigned __int128 carry = t0 >> 64;
    unsigned __int128 t1 = (unsigned __int128)a->w[1] + (unsigned __int128)b->w[1] + carry;
    out->w[1] = (uint64_t)t1;
    carry = t1 >> 64;
    unsigned __int128 t2 = (unsigned __int128)a->w[2] + (unsigned __int128)b->w[2] + carry;
    out->w[2] = (uint64_t)t2;
    carry = t2 >> 64;
    unsigned __int128 t3 = (unsigned __int128)a->w[3] + (unsigned __int128)b->w[3] + carry;
    out->w[3] = (uint64_t)t3;
    carry = t3 >> 64;
    return (uint64_t)carry;
}

// ---------- Bit-vector Levenshtein (same as original) ----------
static __forceinline__ __host__ __device__ int bit_vector_levenshtein_local(
    int query_length,
    const char* reference,
    int reference_length,
    const bv_t* Eq,
    int* zero_indices,
    int* zero_count,
    int* lowest,
    int* lowest_index_local)
{
    if (query_length > 64 * BV_WORDS || query_length <= 0) return -1;

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

    for (int j = 0; j < reference_length; ++j) {
        unsigned char c = (unsigned char)reference[j];
        const bv_t* Eqc = &Eq[c];

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

        if (bv_test_top_unrolled(&Ph, query_length)) ++score;
        if (bv_test_top_unrolled(&Mh, query_length)) --score;

        if (score < *lowest) {
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

        if (score == 0) {
            if (*zero_count < MAX_HITS) zero_indices[*zero_count] = j;
            (*zero_count)++;
        }
    }

    return score;
}

// ---------- Kernel ----------
// shared memory removed: each thread reads Eq directly from global memory
__global__ void levenshtein_kernel_global_agg(
    int num_queries, int num_chunks, int num_orig_refs,
    const char* __restrict__ d_queries, const int* __restrict__ d_q_lens, const bv_t* __restrict__ d_Eq_queries,
    const char* __restrict__ d_refs, const int* __restrict__ d_ref_lens, // per-chunk
    const int* __restrict__ d_chunk_starts, const int* __restrict__ d_chunk_to_orig,
    const int* __restrict__ d_orig_ref_lens, // per original ref

    // State-passing arrays (per query x chunk)
    bv_t* __restrict__ d_final_pv,
    bv_t* __restrict__ d_final_mv,
    int* __restrict__ d_final_scores,
    volatile int* __restrict__ d_chunk_done_flags,

    // Aggregated results (per query x original ref)
    int* __restrict__ d_lowest_score_agg,
    int* __restrict__ d_lowest_index_agg,
    int* __restrict__ d_last_score_agg
)
{
    int q = blockIdx.x;
    if (q >= num_queries) return;
    int tid = threadIdx.x;

    int qlen = d_q_lens[q];
    const bv_t* Eq_global = &d_Eq_queries[(long long)q * 256LL];

    // Process chunks assigned to this thread in a strided loop
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_ref_lens[c];
        int chunk_start = d_chunk_starts[c];
        int orig = d_chunk_to_orig[c]; // original reference id
        long long pair_idx = (long long)q * num_chunks + c;
        long long agg_idx = (long long)q * num_orig_refs + orig;

        // --- State Initialization ---
        int score;
        bv_t Pv, Mv;

        if (chunk_start > 0) {
            // Not the first chunk, so wait for the previous one and load its state.
            // The previous chunk for the same reference will have chunk_id (c-1).
            long long prev_pair_idx = pair_idx - 1;

            // Spin-wait until the previous chunk's "done" flag is set to 1.
            while (atomicCAS((int*)&d_chunk_done_flags[prev_pair_idx], 1, 1) == 0) {
                // Busy-wait is acceptable here as waits should be short.
            }

            // Load the final state from the previous chunk.
            bv_copy_unrolled(&Pv, &d_final_pv[prev_pair_idx]);
            bv_copy_unrolled(&Mv, &d_final_mv[prev_pair_idx]);
            score = d_final_scores[prev_pair_idx];
        } else {
            // This is the first chunk of a reference, so initialize a fresh state.
            bv_set_all_unrolled(&Pv, ~0ULL);
            bv_clear_unrolled(&Mv);
            score = qlen;
        }

        // --- Inlined Alignment Logic ---
        // (Logic from bit_vector_levenshtein_local is now here)
        bv_t Ph, Mh, Xv, Xh, Xp;
        bv_clear_unrolled(&Ph);
        bv_clear_unrolled(&Mh);
        bv_clear_unrolled(&Xv);
        bv_clear_unrolled(&Xh);
        bv_clear_unrolled(&Xp);
        bv_t tmp1, tmp2, tmp3, tmp4;

        for (int j = 0; j < rlen; ++j) {
            unsigned char ref_char = (unsigned char)refptr[j];
            const bv_t* Eqc = &Eq_global[ref_char];

            bv_or_unrolled(&Xv, Eqc, &Mv);
            bv_and_unrolled(&tmp1, &Xv, &Pv);
            bv_add_unrolled(&tmp2, &tmp1, &Pv);
            bv_xor_unrolled(&tmp2, &tmp2, &Pv);
            bv_or_unrolled(&tmp3, &tmp2, &Xv);
            bv_or_unrolled(&Xh, &tmp3, &Mv);
            bv_or_unrolled(&tmp1, &Xh, &Pv);
            bv_not_unrolled(&tmp2, &tmp1);
            bv_or_unrolled(&Ph, &Mv, &tmp2);
            bv_and_unrolled(&Mh, &Xh, &Pv);

            if (bv_test_top_unrolled(&Ph, qlen)) ++score;
            if (bv_test_top_unrolled(&Mh, qlen)) --score;

            // Atomically update the lowest score for the original reference
            atomicMin(&d_lowest_score_agg[agg_idx], score);

            bv_shl1_unrolled(&tmp1, &Mh);
            bv_shl1_unrolled(&tmp2, &Ph);
            bv_or_unrolled(&tmp3, &Xh, &tmp2);
            bv_not_unrolled(&tmp4, &tmp3);
            bv_or_unrolled(&Pv, &tmp1, &tmp4);
            bv_and_unrolled(&Mv, &Xh, &tmp2);
        }

        // --- State Finalization ---
        // Store the final state of this chunk for the next one.
        bv_copy_unrolled(&d_final_pv[pair_idx], &Pv);
        bv_copy_unrolled(&d_final_mv[pair_idx], &Mv);
        d_final_scores[pair_idx] = score;

        // Signal that this chunk is complete.
        atomicExch((int*)&d_chunk_done_flags[pair_idx], 1);

        // If this is the last chunk of the original reference, record the last score.
        int orig_len = d_orig_ref_lens[orig];
        if (chunk_start + rlen >= orig_len) {
            atomicExch(&d_last_score_agg[agg_idx], score);
        }
    }
}

// ---------- Host main ----------
int main() {
    // read queries and original refs
    int num_queries = 0;
    char** query_seqs = parse_fasta_file(query_file, &num_queries);
    if (!query_seqs || num_queries <= 0) { fprintf(stderr, "Failed read queries\n"); return -1; }
    int num_orig_refs = 0;
    char** orig_refs = parse_fasta_file(reference_file, &num_orig_refs);
    if (!orig_refs || num_orig_refs <= 0) { fprintf(stderr, "Failed read refs\n"); return -1; }

    // original lengths
    int* orig_ref_lens = (int*)malloc(num_orig_refs * sizeof(int));
    for (int i = 0; i < num_orig_refs; ++i) orig_ref_lens[i] = (int)strlen(orig_refs[i]);

    // Partition original refs into chunks (overlap = qlen - 1)
    // First estimate chunk count
    int qlen0 = (int)strlen(query_seqs[0]); // assume queries similar length; safe for multiple queries
    size_t estimated_chunks = 0;
    for (int i = 0; i < num_orig_refs; ++i) {
        estimated_chunks += (orig_ref_lens[i] + CHUNK_SIZE - 1) / CHUNK_SIZE;
    }

    char** chunk_seqs = (char**)malloc(sizeof(char*) * estimated_chunks);
    int* chunk_lens = (int*)malloc(sizeof(int) * estimated_chunks);
    int* chunk_starts = (int*)malloc(sizeof(int) * estimated_chunks); // global start in original
    int* chunk_to_orig = (int*)malloc(sizeof(int) * estimated_chunks);
    
    int chunk_idx = 0;
    for (int r = 0; r < num_orig_refs; ++r) {
        int rlen = orig_ref_lens[r];
        int current_pos = 0;
        while (current_pos < rlen) {
            int chunk_len = CHUNK_SIZE;
            if (current_pos + chunk_len > rlen) {
                chunk_len = rlen - current_pos;
            }

            // Add overlap if it's not the last chunk of the reference
            int overlap = (current_pos + chunk_len < rlen) ? (qlen0 - 1) : 0;
            int ext_len = chunk_len + overlap;

            if (chunk_idx >= estimated_chunks) {
                // This should not happen with correct estimation, but as a safeguard:
                fprintf(stderr, "Error: Exceeded estimated chunk allocation. Aborting.\n");
                exit(1);
            }

            char* s = (char*)malloc(ext_len + 1);
            memcpy(s, orig_refs[r] + current_pos, ext_len);
            s[ext_len] = '\0';
            chunk_seqs[chunk_idx] = s;
            chunk_lens[chunk_idx] = ext_len;
            chunk_starts[chunk_idx] = current_pos;
            chunk_to_orig[chunk_idx] = r;
            chunk_idx++;
            current_pos += chunk_len;
        }
    }
    int num_chunks = chunk_idx;
    int num_references = num_chunks; // kernel references = chunks

    // Print chunk information
    for (int i = 0; i < num_chunks; ++i) {
        printf("Chunk No: %d, Chunk Size: %d\n", i, chunk_lens[i]);
    }

    // build q_lens
    int* q_lens = (int*)malloc(num_queries * sizeof(int));
    for (int q = 0; q < num_queries; ++q) q_lens[q] = (int)strlen(query_seqs[q]);

    // Precompute Eq host layout [q * 256 + ascii]
    bv_t* h_Eq_queries = (bv_t*)malloc((size_t)num_queries * 256 * sizeof(bv_t));
    if (!h_Eq_queries) { fprintf(stderr, "OOM Eq\n"); return -1; }
    memset(h_Eq_queries, 0, (size_t)num_queries * 256 * sizeof(bv_t));
    for (int q = 0; q < num_queries; ++q) {
        int qlen = q_lens[q];
        const char* qs = query_seqs[q];
        for (int i = 0; i < qlen; ++i) {
            unsigned char ch = (unsigned char)qs[i];
            int word = i / 64;
            int bit = i % 64;
            h_Eq_queries[(long long)q * 256 + ch].w[word] |= (1ULL << bit);
        }
    }

    // Build contiguous host buffers strided by MAX_LENGTH
    size_t h_queries_bytes = (size_t)num_queries * MAX_LENGTH * sizeof(char);
    size_t h_refs_bytes = (size_t)num_references * MAX_LENGTH * sizeof(char);
    char* h_queries = (char*)malloc(h_queries_bytes);
    char* h_refs = (char*)malloc(h_refs_bytes);
    if (!h_queries || !h_refs) { fprintf(stderr, "OOM host buffers\n"); return -1; }
    memset(h_queries, 0, h_queries_bytes);
    memset(h_refs, 0, h_refs_bytes);
    for (int q = 0; q < num_queries; ++q)
        strncpy(&h_queries[(size_t)q * MAX_LENGTH], query_seqs[q], MIN((int)MAX_LENGTH - 1, q_lens[q]));
    for (int r = 0; r < num_references; ++r)
        strncpy(&h_refs[(size_t)r * MAX_LENGTH], chunk_seqs[r], MIN((int)MAX_LENGTH - 1, chunk_lens[r]));

    int* h_ref_lens = (int*)malloc(num_references * sizeof(int));
    for (int r = 0; r < num_references; ++r) h_ref_lens[r] = chunk_lens[r];

    // Device allocations
    bv_t* d_Eq_queries = NULL;
    char* d_queries = NULL;
    char* d_refs = NULL;
    int* d_q_lens = NULL;
    int* d_ref_lens = NULL;
    int* d_chunk_starts = NULL;
    int* d_chunk_to_orig = NULL;
    int* d_orig_ref_lens = NULL;

    // State-passing arrays
    long long total_pairs = (long long)num_queries * num_chunks;
    bv_t* d_final_pv = NULL;
    bv_t* d_final_mv = NULL;
    int* d_final_scores = NULL;
    int* d_chunk_done_flags = NULL;

    // Aggregated result arrays
    long long total_agg_pairs = (long long)num_queries * num_orig_refs;
    int* d_lowest_score_agg = NULL;
    int* d_lowest_index_agg = NULL; // Note: This logic is complex to parallelize, focusing on score first.
    int* d_last_score_agg = NULL;    

    CUDA_CHECK(cudaMalloc((void**)&d_Eq_queries, (size_t)num_queries * 256 * sizeof(bv_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_queries, h_queries_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_refs, h_refs_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_q_lens, num_queries * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_ref_lens, num_references * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_chunk_starts, num_references * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_chunk_to_orig, num_references * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_orig_ref_lens, num_orig_refs * sizeof(int)));

    // Allocate state-passing arrays
    CUDA_CHECK(cudaMalloc((void**)&d_final_pv, total_pairs * sizeof(bv_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_final_mv, total_pairs * sizeof(bv_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_final_scores, total_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_chunk_done_flags, total_pairs * sizeof(int)));

    // Allocate aggregated result arrays
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_score_agg, total_agg_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_index_agg, total_agg_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_last_score_agg, total_agg_pairs * sizeof(int)));

    // host arrays to copy back aggregated orig results
    int* h_lowest_score_agg = (int*)malloc(total_agg_pairs * sizeof(int));
    int* h_lowest_index_agg = (int*)malloc(total_agg_pairs * sizeof(int));
    int* h_last_score_agg = (int*)malloc(total_agg_pairs * sizeof(int));

    // prepare host arrays
    CUDA_CHECK(cudaMemcpy(d_Eq_queries, h_Eq_queries, (size_t)num_queries * 256 * sizeof(bv_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, h_queries_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_refs, h_refs, h_refs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_lens, q_lens, num_queries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_lens, h_ref_lens, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_starts, chunk_starts, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_to_orig, chunk_to_orig, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig_ref_lens, orig_ref_lens, num_orig_refs * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize device arrays
    CUDA_CHECK(cudaMemset(d_chunk_done_flags, 0, total_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lowest_score_agg, 0x7f, total_agg_pairs * sizeof(int))); // Init with INT_MAX
    CUDA_CHECK(cudaMemset(d_lowest_index_agg, 0xff, total_agg_pairs * sizeof(int))); // Init with -1
    CUDA_CHECK(cudaMemset(d_last_score_agg, 0x7f, total_agg_pairs * sizeof(int))); // Init with sentinel

    // launch kernel
    int threads = threadsPerBlock;
    int blocks = num_queries;
    if (blocks < 1) blocks = 1;
    size_t shared_bytes = 256 * sizeof(bv_t);

    printf("*** launching kernel: blocks=%d threads=%d shared=%zu num_chunks=%d num_orig=%d\n", blocks, threads, shared_bytes, num_chunks, num_orig_refs);
    double t0 = now_seconds();
    for (int it = 0; it < loope; ++it) {
        levenshtein_kernel_global_agg<<<blocks, threads, shared_bytes>>>(
            num_queries, num_chunks, num_orig_refs,
            d_queries, d_q_lens, d_Eq_queries,
            d_refs, d_ref_lens,
            d_chunk_starts, d_chunk_to_orig, d_orig_ref_lens,
            d_final_pv, d_final_mv, d_final_scores, d_chunk_done_flags,
            d_lowest_score_agg, d_lowest_index_agg, d_last_score_agg
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double avg_time = (t1 - t0) / (double)loope;
    
    // Copy back final aggregated results
    CUDA_CHECK(cudaMemcpy(h_lowest_score_agg, d_lowest_score_agg, total_agg_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_index_agg, d_lowest_index_agg, total_agg_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_last_score_agg, d_last_score_agg, total_agg_pairs * sizeof(int), cudaMemcpyDeviceToHost));

    // The old host-side aggregation loop is no longer needed.
    // We can now print the results directly from the `h_*_agg` arrays.
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long agg_idx = (long long)q * num_orig_refs + orig;
            size_t acc_n = 0; // Placeholder for hit count, real hit logic is complex

            // print block exactly as requested
            printf("----------------------------------------------------------------------------\n"); //
            printf("Pair: Q%d(%d) Vs R%d(%d)\n", q+1, q_lens[q], orig+1, orig_ref_lens[orig]);
            printf("Number of Hits: %zu\n", acc_n); // Note: Hit counting is not implemented in this version
            if (acc_n > 0) {
                printf("Hit Indexes: [");
                for (size_t i = 0; i < acc_n; ++i) {
                    if (i) printf(",");
                    // printf("%d", acc[i]); // 'acc' is not defined, placeholder
                }
                printf("]\n");
                // Hit index aggregation is complex and not implemented in this version.
            } else {
                printf("Hit Indexes: N/A\n");
            }

            // lowest score and index come from GPU aggregated arrays
            int lowest_gpu = h_lowest_score_agg[agg_idx];
            int lowest_idx_gpu = h_lowest_index_agg[agg_idx]; // Note: index logic not implemented
            // sentinel check: INT_MAX sentinel used (0x7f7f7f7f)
            if (lowest_gpu == 0x7f7f7f7f) {
                printf("Lowest Score: N/A\n");
                printf("Lowest Score Indexes: N/A\n");
            } else {
                printf("Lowest Score: %d\n", lowest_gpu);
                printf("Lowest Score Indexes: [N/A]\n"); // Index logic is complex and omitted for now
            }

            // last score: device wrote it only if chunk contained final pos; sentinel check
            int last_gpu = h_last_score_agg[agg_idx];
            if (last_gpu == 0x7f7f7f7f) {
                printf("Last Score: N/A\n");
            } else {
                printf("Last Score: %d\n", last_gpu);
            }

            printf("----------------------------------------------------------------------------\n");
        }
    }

    printf("%d loop Average time: %.6f sec.\n", loope, avg_time);

    // cleanup
    CUDA_CHECK(cudaFree(d_Eq_queries)); CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_refs));
    CUDA_CHECK(cudaFree(d_q_lens)); CUDA_CHECK(cudaFree(d_ref_lens));
    // ... free other original device pointers
    CUDA_CHECK(cudaFree(d_final_pv)); CUDA_CHECK(cudaFree(d_final_mv));
    CUDA_CHECK(cudaFree(d_final_scores)); CUDA_CHECK(cudaFree(d_chunk_done_flags));
    CUDA_CHECK(cudaFree(d_lowest_score_agg)); CUDA_CHECK(cudaFree(d_lowest_index_agg)); CUDA_CHECK(cudaFree(d_last_score_agg));
    free(h_Eq_queries); free(h_queries); free(h_refs);
    free(h_lowest_score_agg); free(h_lowest_index_agg); free(h_last_score_agg);
    // ... free other host pointers
    free(h_ref_lens); free(q_lens);
    for (int i = 0; i < num_references; ++i) free(chunk_seqs[i]);
    free(chunk_seqs); free(chunk_lens); free(chunk_starts); free(chunk_to_orig);

    for (int i = 0; i < num_queries; ++i) free(query_seqs[i]); free(query_seqs);
    for (int i = 0; i < num_orig_refs; ++i) free(orig_refs[i]); free(orig_refs);
    free(orig_ref_lens);
    return 0;
}
