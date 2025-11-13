// hycuda.cu
// Hyyrö bit-vector Levenshtein with GPU-side aggregation using partition_utils.h
// Compile:
//   nvcc -O3 unified_leven_partitioned_gpu.cu C_utils.c -o unified_leven_partitioned_gpu
//
// Run:
//   ./unified_leven_partitioned_gpu

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "C_utils.h"
#include "partition.h"
#include "loading.h"
#include "sender.h"
#include <pthread.h>



#define MAX_LENGTH (1 << 24)   // per-slot buffer size (must be >= longest chunk)
#define MAX_HITS 1024

// ============= USER PARAMETERS =============
#define threadsPerBlock 256
#define loope 10
#define CHUNK_SIZE 166700
#define PARTITION_THRESHOLD 1000000
// ===========================================

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

// ---------- Bit-vector Levenshtein (returns local lowest index too) ----------
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
// Each block = a query. Each thread = strided chunks (where "reference" is a chunk)
__global__ void levenshtein_kernel_shared_agg(
    int num_queries, int num_chunks, int num_orig_refs,
    const char* __restrict__ d_queries, const int* __restrict__ d_q_lens, const bv_t* __restrict__ d_Eq_queries,
    const char* __restrict__ d_refs, const int* __restrict__ d_ref_lens, // per-chunk
    const int* __restrict__ d_chunk_starts, const int* __restrict__ d_chunk_to_orig,
    const int* __restrict__ d_orig_ref_lens, // per original ref
    // pair-level outputs (per query x chunk)
    int* __restrict__ d_pair_distances, int* __restrict__ d_pair_zcounts, int* __restrict__ d_pair_zindices,
    // per-original aggregated (per query x original ref)
    int* __restrict__ d_lowest_score_orig, int* __restrict__ d_lowest_index_orig, int* __restrict__ d_last_score_orig
)
{
    extern __shared__ bv_t s_Eq[]; // 256 entries
    int q = blockIdx.x;
    if (q >= num_queries) return;
    int tid = threadIdx.x;

    // load Eq for this query into shared
    for (int i = tid; i < 256; i += blockDim.x) {
        s_Eq[i] = d_Eq_queries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int qlen = d_q_lens[q];
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
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
        d_pair_zcounts[pair_idx] = local_zero_count;
        // convert local zero positions to global coordinates and write into pair zindices
        long long base_zptr = pair_idx * MAX_HITS;
        for (int k = 0; k < local_zero_count && k < MAX_HITS; ++k) {
            int global_pos = chunk_start + local_zero_indices[k];
            d_pair_zindices[base_zptr + k] = global_pos;
        }
        // fill remaining with -1 for safety (optional)
        for (int k = local_zero_count; k < MAX_HITS; ++k) d_pair_zindices[base_zptr + k] = -1;

        // -------- GPU-side aggregation for original reference (per query x orig) --------
        long long orig_pair_idx = (long long)q * num_orig_refs + orig;

        // 1) update lowest score and lowest index atomically (atomic CAS loop)
        // Only attempt if local_lowest_val is valid
        if (local_lowest_idx >= 0) {
            int global_lowest_pos = chunk_start + local_lowest_idx;
            // atomicCAS loop for lowest score update
            int old;
            do {
                old = d_lowest_score_orig[orig_pair_idx];
                if (local_lowest_val < old) { // attempt to replace
                    int res = atomicCAS(&d_lowest_score_orig[orig_pair_idx], old, local_lowest_val);
                    if (res == old) {
                        // we succeeded in replacing lowest score, now set index
                        atomicExch(&d_lowest_index_orig[orig_pair_idx], global_lowest_pos);
                        break;
                    } else {
                        // someone else changed it; loop to re-check
                    }
                } else {
                    break; // no update needed
                }
            } while (true);
        }
    }
}

typedef struct {
    int num_queries;
    int num_orig_refs;
    char **query_seqs;
    char **orig_refs;
} GPUArgs;

void* run_hyyro_gpu(void* args) {
    GPUArgs* gpu_args = (GPUArgs*)args;
    int num_queries = gpu_args->num_queries;
    int num_orig_refs = gpu_args->num_orig_refs;
    char** query_seqs = gpu_args->query_seqs;
    char** orig_refs = gpu_args->orig_refs;

    // Original reference lengths
    int* orig_ref_lens = (int*)malloc(num_orig_refs * sizeof(int));
    for (int i = 0; i < num_orig_refs; ++i) orig_ref_lens[i] = (int)strlen(orig_refs[i]);
    for (int i = 0; i < num_orig_refs; ++i) 
    printf("orig_ref_lens[%d] = %d\n", i, orig_ref_lens[i]);

    // Partition original refs for GPU
    int qlen0 = (int)strlen(query_seqs[0]);
    int overlap = qlen0 - 1;
    PartitionedRefs part_refs = partition_references(orig_refs, orig_ref_lens, num_orig_refs,
                                                     overlap, CHUNK_SIZE, PARTITION_THRESHOLD);
    int num_chunks = part_refs.num_chunks;
    int num_references = num_chunks;

    printf("Partitioned into %d chunks\n\n", num_chunks);

    // Build query lengths
    int* q_lens = (int*)malloc(num_queries * sizeof(int));
    for (int q = 0; q < num_queries; ++q) q_lens[q] = (int)strlen(query_seqs[q]);

    // Precompute Eq host layout
    bv_t* h_Eq_queries = (bv_t*)malloc((size_t)num_queries * 256 * sizeof(bv_t));
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

    // Build contiguous host buffers
    size_t h_queries_bytes = (size_t)num_queries * MAX_LENGTH * sizeof(char);
    size_t h_refs_bytes = (size_t)num_references * MAX_LENGTH * sizeof(char);
    char* h_queries = (char*)malloc(h_queries_bytes);
    char* h_refs = (char*)malloc(h_refs_bytes);
    memset(h_queries, 0, h_queries_bytes);
    memset(h_refs, 0, h_refs_bytes);
    for (int q = 0; q < num_queries; ++q)
        strncpy(&h_queries[(size_t)q * MAX_LENGTH], query_seqs[q], MIN((int)MAX_LENGTH - 1, q_lens[q]));
    for (int r = 0; r < num_references; ++r)
        strncpy(&h_refs[(size_t)r * MAX_LENGTH], part_refs.chunk_seqs[r], MIN((int)MAX_LENGTH - 1, part_refs.chunk_lens[r]));

    int* h_ref_lens = (int*)malloc(num_references * sizeof(int));
    for (int r = 0; r < num_references; ++r) h_ref_lens[r] = part_refs.chunk_lens[r];

    // Build orig->chunk mapping
    int* orig_chunk_counts = NULL;
    int** orig_chunk_lists = NULL;
    build_orig_to_chunk_mapping(&part_refs, num_orig_refs, &orig_chunk_counts, &orig_chunk_lists);

    // Device allocations
    bv_t* d_Eq_queries; char* d_queries; char* d_refs;
    int *d_q_lens, *d_ref_lens, *d_chunk_starts, *d_chunk_to_orig, *d_orig_ref_lens;
    long long total_pair_chunks = (long long)num_queries * (long long)num_references;
    int *d_pair_distances, *d_pair_zcounts, *d_pair_zindices;
    int *d_lowest_score_orig, *d_lowest_index_orig, *d_last_score_orig;

    CUDA_CHECK(cudaMalloc((void**)&d_Eq_queries, (size_t)num_queries * 256 * sizeof(bv_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_queries, h_queries_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_refs, h_refs_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_q_lens, num_queries * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_ref_lens, num_references * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_chunk_starts, num_references * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_chunk_to_orig, num_references * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_orig_ref_lens, num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_pair_distances, (size_t)total_pair_chunks * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_pair_zcounts, (size_t)total_pair_chunks * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_pair_zindices, (size_t)total_pair_chunks * MAX_HITS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_index_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_last_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));

    int* h_lowest_score_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));
    int* h_lowest_index_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));
    int* h_last_score_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));

    CUDA_CHECK(cudaMemcpy(d_Eq_queries, h_Eq_queries, (size_t)num_queries * 256 * sizeof(bv_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, h_queries_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_refs, h_refs, h_refs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_lens, q_lens, num_queries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_lens, h_ref_lens, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_starts, part_refs.chunk_starts, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_to_orig, part_refs.chunk_to_orig, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig_ref_lens, orig_ref_lens, num_orig_refs * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_lowest_score_orig, 0x7f, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lowest_index_orig, 0xff, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_last_score_orig, 0x7f, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pair_distances, 0xff, (size_t)total_pair_chunks * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pair_zcounts, 0, (size_t)total_pair_chunks * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pair_zindices, 0xff, (size_t)total_pair_chunks * MAX_HITS * sizeof(int)));

    // Launch kernel
    int threads = threadsPerBlock;
    int blocks = num_queries < 1 ? 1 : num_queries;
    size_t shared_bytes = 256 * sizeof(bv_t);

    double t0 = now_seconds();
    for (int it = 0; it < loope; ++it) {
        levenshtein_kernel_shared_agg<<<blocks, threads, shared_bytes>>>(
            num_queries, num_chunks, num_orig_refs,
            d_queries, d_q_lens, d_Eq_queries,
            d_refs, d_ref_lens,
            d_chunk_starts, d_chunk_to_orig,
            d_orig_ref_lens,
            d_pair_distances, d_pair_zcounts, d_pair_zindices,
            d_lowest_score_orig, d_lowest_index_orig, d_last_score_orig
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double avg_time = (t1 - t0) / (double)loope;

    // Copy back results
    long long pairs = total_pair_chunks;
    int* h_pair_distances = (int*)malloc((size_t)pairs * sizeof(int));
    int* h_pair_zcounts = (int*)malloc((size_t)pairs * sizeof(int));
    int* h_pair_zindices = (int*)malloc((size_t)pairs * MAX_HITS * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_pair_distances, d_pair_distances, (size_t)pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zcounts, d_pair_zcounts, (size_t)pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zindices, d_pair_zindices, (size_t)pairs * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_score_orig, d_lowest_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_index_orig, d_lowest_index_orig, (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_last_score_orig, d_last_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));

    // Correct last scores as in original main
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long orig_pair_idx = (long long)q * num_orig_refs + orig;
            int orig_len = orig_ref_lens[orig];
            int correct_last_score = 0x7f7f7f7f;
            for (int ci_idx = 0; ci_idx < orig_chunk_counts[orig]; ++ci_idx) {
                int chunk_id = orig_chunk_lists[orig][ci_idx];
                int chunk_start = part_refs.chunk_starts[chunk_id];
                int chunk_len = part_refs.chunk_lens[chunk_id];
                if (chunk_start + chunk_len == orig_len) {
                    long long pair_idx = (long long)q * num_chunks + chunk_id;
                    correct_last_score = h_pair_distances[pair_idx];
                    break;
                }
            }
            if (correct_last_score != 0x7f7f7f7f) h_last_score_orig[orig_pair_idx] = correct_last_score;
        }
    }

    // Aggregate and print results
    printf("\n=== Results ===\n");
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            size_t cap = 4096;
            int *acc = (int*)malloc(sizeof(int) * cap);
            size_t acc_n = 0;
            int min_dist = INT_MAX;
            for (int ci_idx = 0; ci_idx < orig_chunk_counts[orig]; ++ci_idx) {
                int chunk_id = orig_chunk_lists[orig][ci_idx];
                long long pair_idx = (long long)q * num_chunks + chunk_id;
                int zc = h_pair_zcounts[pair_idx];
                int dist = h_pair_distances[pair_idx];
                if (dist < min_dist) min_dist = dist;
                for (int k = 0; k < zc && k < MAX_HITS; ++k) {
                    int val = h_pair_zindices[pair_idx * MAX_HITS + k];
                    if (val >= 0) {
                        if (acc_n >= cap) { cap *= 2; acc = (int*)realloc(acc, sizeof(int) * cap); }
                        acc[acc_n++] = val;
                    }
                }
            }
            if (acc_n > 0) {
                qsort(acc, acc_n, sizeof(int), compare_ints);
                size_t write = 1;
                for (size_t i = 1; i < acc_n; ++i) if (acc[i] != acc[write-1]) acc[write++] = acc[i];
                acc_n = write;
            }

            printf("----------------------------------------------------------------------------\n");
            printf("Pair: Q%d(%d) Vs R%d(%d)\n", q+1, q_lens[q], orig+1, orig_ref_lens[orig]);
            printf("Number of Hits: %zu\n", acc_n);
            if (acc_n > 0) {
                printf("Hit Indexes: [");
                for (size_t i = 0; i < acc_n; ++i) { if (i) printf(","); printf("%d", acc[i]); }
                printf("]\n");
            } else { printf("Hit Indexes: N/A\n"); }

            int lowest_gpu = h_lowest_score_orig[(long long)q * num_orig_refs + orig];
            int lowest_idx_gpu = h_lowest_index_orig[(long long)q * num_orig_refs + orig];
            if (lowest_gpu == 0x7f7f7f7f) { printf("Lowest Score: N/A\nLowest Score Indexes: N/A\n"); }
            else {
                printf("Lowest Score: %d\n", lowest_gpu);
                if (lowest_idx_gpu >= 0) printf("Lowest Score Indexes: [%d]\n", lowest_idx_gpu);
                else printf("Lowest Score Indexes: N/A\n");
            }

            int last_gpu = h_last_score_orig[(long long)q * num_orig_refs + orig];
            if (last_gpu == 0x7f7f7f7f) printf("Last Score: N/A\n");
            else printf("Last Score: %d\n", last_gpu);
            printf("----------------------------------------------------------------------------\n");

            free(acc);
        }
    }

    printf("\n%d loop Average time: %.6f sec.\n", loope, avg_time);

    // Cleanup
    CUDA_CHECK(cudaFree(d_Eq_queries)); CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_refs));
    CUDA_CHECK(cudaFree(d_q_lens)); CUDA_CHECK(cudaFree(d_ref_lens));
    CUDA_CHECK(cudaFree(d_chunk_starts)); CUDA_CHECK(cudaFree(d_chunk_to_orig)); CUDA_CHECK(cudaFree(d_orig_ref_lens));
    CUDA_CHECK(cudaFree(d_pair_distances)); CUDA_CHECK(cudaFree(d_pair_zcounts)); CUDA_CHECK(cudaFree(d_pair_zindices));
    CUDA_CHECK(cudaFree(d_lowest_score_orig)); CUDA_CHECK(cudaFree(d_lowest_index_orig)); CUDA_CHECK(cudaFree(d_last_score_orig));

    free(h_Eq_queries); free(h_queries); free(h_refs);
    free(h_pair_distances); free(h_pair_zcounts); free(h_pair_zindices);
    free(h_lowest_score_orig); free(h_lowest_index_orig); free(h_last_score_orig);
    free(h_ref_lens); free(q_lens);
    free(orig_ref_lens);
    free_orig_to_chunk_mapping(orig_chunk_counts, orig_chunk_lists, num_orig_refs);
    free_partitioned_refs(&part_refs);

    return NULL;
}



// ---------- Host main ----------

int main() {
    printf("=== Hyyrö Bit-Vector Levenshtein with Partitioning ===\n");
    printf("Loading queries and references from memory using loading.h\n");
    printf("Chunk size: %d\n", CHUNK_SIZE);
    printf("Partition threshold: %d\n", PARTITION_THRESHOLD);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Loop iterations: %d\n\n", loope);

    int num_queries = 0;
    int num_orig_refs = 0;

    // Load queries and references
    char** query_seqs = load_queries(&num_queries);
    char **gpu_refs = NULL;    // GPU in-memory sequences
    int *gpu_ref_lens = NULL;
    char** orig_refs = load_references_gpu_fpga(&num_orig_refs, &gpu_refs, &gpu_ref_lens);

    printf("Loaded %d queries and %d references\n", num_queries, num_orig_refs);

    // Prepare arguments for GPU thread
    GPUArgs gpu_args = {
        .num_queries = num_queries,
        .num_orig_refs = num_orig_refs,
        .query_seqs = query_seqs,
        .orig_refs = gpu_refs
    };

    // Create GPU thread
    pthread_t gpu_thread;
    pthread_create(&gpu_thread, NULL, run_hyyro_gpu, &gpu_args);

    // Run FPGA in main thread
    send_and_run_fpga();

    // Wait for GPU thread to finish
    pthread_join(gpu_thread, NULL);

    // Cleanup queries/references
    for (int i = 0; i < num_queries; ++i) free(query_seqs[i]);
    free(query_seqs);
    for (int i = 0; i < num_orig_refs; ++i) free(orig_refs[i]);
    free(orig_refs);

    return 0;
}