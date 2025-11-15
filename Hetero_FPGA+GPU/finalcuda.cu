// unified_leven_partitioned_gpu.cu
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
static __forceinline__ __host__ __device__ void bv_shr1_unrolled(bv_t* out, const bv_t* in) {
    uint64_t c3 = in->w[3] & 1ULL;
    uint64_t c2 = in->w[2] & 1ULL;
    uint64_t c1 = in->w[1] & 1ULL;
    out->w[3] = (in->w[3] >> 1);
    out->w[2] = (in->w[2] >> 1) | (c3 << 63);
    out->w[1] = (in->w[1] >> 1) | (c2 << 63);
    out->w[0] = (in->w[0] >> 1) | (c1 << 63);
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

// mask bits above m-1 to zero (so bits >= m are cleared)
static __forceinline__ __host__ __device__ void bv_mask_top(bv_t *v, int m) {
    if (m >= 64 * BV_WORDS) return; // no masking needed
    int last_word = (m - 1) / 64;
    int last_bit = (m - 1) % 64;
    uint64_t last_mask = (last_bit == 63) ? ~0ULL : ((1ULL << (last_bit + 1)) - 1ULL);

    for (int i = last_word + 1; i < BV_WORDS; ++i) v->w[i] = 0ULL;
    v->w[last_word] &= last_mask;
}

// ---------- Bit-vector Levenshtein (from hycuda.cu) ----------
static __forceinline__ __host__ __device__ int bit_vector_levenshtein_local(
    int query_length,
    const char* reference,
    int reference_length,
    const bv_t* Eq,
    int* zero_indices,
    int* zero_count,
    int* lowest_score,
    int* lowest_indices,
    int* lowest_count)
{
    if (query_length > 64 * BV_WORDS || query_length <= 0) return -1;

    bv_t Pv, Mv, Xv, Xh, Ph, Mh;
    bv_t PrevEqMask, trans_mask, tmp, addtmp, t1, t2;

    bv_set_all_unrolled(&Pv, ~0ULL);
    bv_clear_unrolled(&Mv);
    bv_clear_unrolled(&PrevEqMask);
    bv_clear_unrolled(&trans_mask);

    bv_mask_top(&Pv, query_length);

    *zero_count = 0;
    *lowest_count = 0;
    int score = query_length;
    *lowest_score = score;

    bool have_prev = false;

    for (int j = 0; j < reference_length; ++j) {
        unsigned char c = (unsigned char)reference[j];
        const bv_t* Eqc = &Eq[c];

        // Standard Myers/Hyyrö
        bv_or_unrolled(&Xv, Eqc, &Mv);

        bv_and_unrolled(&tmp, &Xv, &Pv);
        bv_add_unrolled(&addtmp, &tmp, &Pv);
        bv_xor_unrolled(&Xh, &addtmp, &Pv);
        bv_or_unrolled(&Xh, &Xh, &Xv);
        bv_or_unrolled(&Xh, &Xh, &Mv);

        bv_or_unrolled(&tmp, &Xh, &Pv);
        bv_not_unrolled(&t1, &tmp);
        bv_or_unrolled(&Ph, &Mv, &t1);

        bv_and_unrolled(&Mh, &Pv, &Xh);

        if (bv_test_top_unrolled(&Ph, query_length)) ++score;
        if (bv_test_top_unrolled(&Mh, query_length)) --score;

        if (score < *lowest_score) {
            *lowest_score = score;
            *lowest_count = 0;
            // Fall-through to add the first index
        }
        if (score == *lowest_score) {
            if (*lowest_count < MAX_HITS) {
                lowest_indices[*lowest_count] = j;
                (*lowest_count)++;
            }
        }

        if (have_prev) {
            bv_shr1_unrolled(&t1, &PrevEqMask);
            bv_and_unrolled(&trans_mask, Eqc, &t1);
        } else {
            bv_clear_unrolled(&trans_mask);
        }

        bv_not_unrolled(&t2, &trans_mask);
        bv_and_unrolled(&Pv, &Pv, &t2);
        bv_or_unrolled(&Mv, &Mv, &trans_mask);

        bv_shl1_unrolled(&t1, &Mh);
        bv_shl1_unrolled(&t2, &Ph);
        bv_or_unrolled(&tmp, &Xh, &t2);
        bv_not_unrolled(&addtmp, &tmp);
        bv_or_unrolled(&Pv, &t1, &addtmp);

        bv_and_unrolled(&Mv, &Xh, &t2);

        bv_mask_top(&Pv, query_length);
        bv_mask_top(&Mv, query_length);

        bv_copy_unrolled(&PrevEqMask, Eqc);
        have_prev = true;

        if (score == 0) {
            if (*zero_count < MAX_HITS) {
                zero_indices[*zero_count] = j;
                (*zero_count)++;
            }
        }
    }

    return score;
}

// ---------- Kernel ----------
// Each block = a query. Each thread = strided chunks (where "reference" is a chunk)
__global__ void levenshtein_kernel_shared_agg(
    int num_queries, int num_chunks, int num_orig_refs,
    const char* __restrict__ d_queries, const int* __restrict__ d_q_lens, const bv_t* __restrict__ d_Eq_queries,
    const char* __restrict__ d_refs, const int* __restrict__ d_ref_lens,
    const int* __restrict__ d_chunk_starts, const int* __restrict__ d_chunk_to_orig,
    const int* __restrict__ d_orig_ref_lens,
    int* __restrict__ d_pair_distances, int* __restrict__ d_pair_zcounts, int* __restrict__ d_pair_zindices,
    int* __restrict__ d_lowest_score_orig, int* __restrict__ d_lowest_count_orig, int* __restrict__ d_lowest_indices_orig,
    int* __restrict__ d_last_score_orig
)
{
    extern __shared__ bv_t s_Eq[];
    int q = blockIdx.x;
    if (q >= num_queries) return;
    int tid = threadIdx.x;

    for (int i = tid; i < 256; i += blockDim.x) {
        s_Eq[i] = d_Eq_queries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int qlen = d_q_lens[q];
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_ref_lens[c];
        int chunk_start = d_chunk_starts[c];
        int orig = d_chunk_to_orig[c];
        long long pair_idx = (long long)q * num_chunks + c;

        int local_zero_indices[MAX_HITS];
        int local_zero_count = 0;
        int local_lowest_score = INT_MAX;
        int local_lowest_indices[MAX_HITS];
        int local_lowest_count = 0;

        int dist = bit_vector_levenshtein_local(qlen, refptr, rlen, s_Eq,
            local_zero_indices, &local_zero_count, 
            &local_lowest_score, local_lowest_indices, &local_lowest_count);

        // Write pair-level results
        d_pair_distances[pair_idx] = dist;
        d_pair_zcounts[pair_idx] = local_zero_count;
        long long base_zptr = pair_idx * MAX_HITS;
        for (int k = 0; k < local_zero_count && k < MAX_HITS; ++k) {
            int global_pos = chunk_start + local_zero_indices[k];
            d_pair_zindices[base_zptr + k] = global_pos;
        }
        for (int k = local_zero_count; k < MAX_HITS; ++k) {
            d_pair_zindices[base_zptr + k] = -1;
        }

        // -------- GPU-side aggregation with ALL lowest indexes --------
        long long orig_pair_idx = (long long)q * num_orig_refs + orig;
        long long orig_indices_base = orig_pair_idx * MAX_HITS;

        if (local_lowest_count > 0) {
            // Try to update the lowest score atomically
            int old_score = atomicMin(&d_lowest_score_orig[orig_pair_idx], local_lowest_score);
            
            // Now add all our lowest indexes if they match the current best
            int current_best = d_lowest_score_orig[orig_pair_idx];
            
            if (local_lowest_score == current_best) {
                // Add all our local_lowest_indices to the global array
                for (int k = 0; k < local_lowest_count; ++k) {
                    int global_lowest_pos = chunk_start + local_lowest_indices[k];
                    
                    // Atomically get a slot in the indices array
                    int slot = atomicAdd(&d_lowest_count_orig[orig_pair_idx], 1);
                    
                    if (slot < MAX_HITS) {
                        d_lowest_indices_orig[orig_indices_base + slot] = global_lowest_pos;
                    }
                }
            }
        }
    }
}

typedef struct {
    // Input parameters
    int num_queries;
    int num_orig_refs;
    char **query_seqs;
    char **orig_refs;
    
    // Output parameters - ADD THESE
    int* output_lowest_scores;      // Array: num_queries * num_orig_refs
    int** output_lowest_indices;    // Array of arrays for each query-ref pair
    int* output_lowest_counts;      // Array: num_queries * num_orig_refs
    int* output_last_scores;        // Array: num_queries * num_orig_refs
    int** output_hit_indices;       // Array of arrays for each query-ref pair
    int* output_hit_counts;         // Array: num_queries * num_orig_refs
    double avg_execution_time;
    int success;                    // 1 = success, 0 = failure
} GPUArgs;

void* run_hyyro_gpu(void* args) {
    GPUArgs* gpu_args = (GPUArgs*)args;
    int num_queries = gpu_args->num_queries;
    int num_orig_refs = gpu_args->num_orig_refs;
    char** query_seqs = gpu_args->query_seqs;
    char** orig_refs = gpu_args->orig_refs;

// ---------- Host main ----------
    printf("Chunk size: %d\n", CHUNK_SIZE);
    printf("Partition threshold: %d\n", PARTITION_THRESHOLD);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Loop iterations: %d\n\n", loope);

    // original lengths
    int* orig_ref_lens = (int*)malloc(num_orig_refs * sizeof(int));
    for (int i = 0; i < num_orig_refs; ++i) orig_ref_lens[i] = (int)strlen(orig_refs[i]);

    // Partition original refs into chunks using partition_utils.h
    int qlen0 = (int)strlen(query_seqs[0]); // assume queries similar length
    int overlap = qlen0 - 1;

    PartitionedRefs part_refs = partition_references(
        orig_refs, orig_ref_lens, num_orig_refs,
        overlap, CHUNK_SIZE, PARTITION_THRESHOLD
    );

    int num_chunks = part_refs.num_chunks;
    int num_references = num_chunks; // kernel references = chunks

    printf("Partitioned into %d chunks\n\n", num_chunks);

    // build q_lens
    int* q_lens = (int*)malloc(num_queries * sizeof(int));
    for (int q = 0; q < num_queries; ++q) q_lens[q] = (int)strlen(query_seqs[q]);

    // Precompute Eq host layout [q * 256 + ascii]
    bv_t* h_Eq_queries = (bv_t*)malloc((size_t)num_queries * 256 * sizeof(bv_t));
    if (!h_Eq_queries) { fprintf(stderr, "OOM Eq\n"); return NULL; }
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
    if (!h_queries || !h_refs) { fprintf(stderr, "OOM host buffers\n"); return NULL; }
    memset(h_queries, 0, h_queries_bytes);
    memset(h_refs, 0, h_refs_bytes);
    for (int q = 0; q < num_queries; ++q)
        strncpy(&h_queries[(size_t)q * MAX_LENGTH], query_seqs[q], MIN((int)MAX_LENGTH - 1, q_lens[q]));
    for (int r = 0; r < num_references; ++r)
        strncpy(&h_refs[(size_t)r * MAX_LENGTH], part_refs.chunk_seqs[r], MIN((int)MAX_LENGTH - 1, part_refs.chunk_lens[r]));

    int* h_ref_lens = (int*)malloc(num_references * sizeof(int));
    for (int r = 0; r < num_references; ++r) h_ref_lens[r] = part_refs.chunk_lens[r];

    // build orig->chunk list counts for host aggregation
    int* orig_chunk_counts = NULL;
    int** orig_chunk_lists = NULL;
    build_orig_to_chunk_mapping(&part_refs, num_orig_refs, &orig_chunk_counts, &orig_chunk_lists);

    // Device allocations
    bv_t* d_Eq_queries = NULL;
    char* d_queries = NULL;
    char* d_refs = NULL;
    int* d_q_lens = NULL;
    int* d_ref_lens = NULL;
    int* d_chunk_starts = NULL;
    int* d_chunk_to_orig = NULL;
    int* d_orig_ref_lens = NULL;

    long long total_pair_chunks = (long long)num_queries * (long long)num_references;
    int* d_pair_distances = NULL;
    int* d_pair_zcounts = NULL;
    int* d_pair_zindices = NULL;

    // per-original aggregated arrays
    int* d_lowest_score_orig = NULL;
    int* d_lowest_count_orig = NULL;  // NEW: count of lowest indexes
    int* d_lowest_indices_orig = NULL; // NEW: array of all lowest indexes
    int* d_last_score_orig = NULL;

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
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_count_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_indices_orig, (size_t)num_queries * num_orig_refs * MAX_HITS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_last_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));

    // host arrays to copy back aggregated orig results
    int* h_lowest_score_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));
    int* h_lowest_count_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));
    int* h_lowest_indices_orig = (int*)malloc((size_t)num_queries * num_orig_refs * MAX_HITS * sizeof(int));
    int* h_last_score_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));

    // prepare host arrays
    CUDA_CHECK(cudaMemcpy(d_Eq_queries, h_Eq_queries, (size_t)num_queries * 256 * sizeof(bv_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, h_queries_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_refs, h_refs, h_refs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_lens, q_lens, num_queries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_lens, h_ref_lens, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_starts, part_refs.chunk_starts, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_to_orig, part_refs.chunk_to_orig, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig_ref_lens, orig_ref_lens, num_orig_refs * sizeof(int), cudaMemcpyHostToDevice));

    // init per-original arrays on device
    CUDA_CHECK(cudaMemset(d_lowest_score_orig, 0x7f, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lowest_count_orig, 0, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lowest_indices_orig, 0xff, (size_t)num_queries * num_orig_refs * MAX_HITS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_last_score_orig, 0x7f, (size_t)num_queries * num_orig_refs * sizeof(int)));

    // init pair-level arrays
    CUDA_CHECK(cudaMemset(d_pair_distances, 0xff, (size_t)total_pair_chunks * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pair_zcounts, 0, (size_t)total_pair_chunks * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pair_zindices, 0xff, (size_t)total_pair_chunks * MAX_HITS * sizeof(int)));

    // launch kernel
    int threads = threadsPerBlock;
    int blocks = num_queries;
    if (blocks < 1) blocks = 1;
    size_t shared_bytes = 256 * sizeof(bv_t);

    printf("*** launching kernel: blocks=%d threads=%d shared=%zu num_chunks=%d num_orig=%d\n",
           blocks, threads, shared_bytes, num_chunks, num_orig_refs);

    double t0 = now_seconds();
    for (int it = 0; it < loope; ++it) {
        levenshtein_kernel_shared_agg<<<blocks, threads, shared_bytes>>>(
            num_queries, num_chunks, num_orig_refs,
            d_queries, d_q_lens, d_Eq_queries,
            d_refs, d_ref_lens,
            d_chunk_starts, d_chunk_to_orig,
            d_orig_ref_lens,
            d_pair_distances, d_pair_zcounts, d_pair_zindices,
            d_lowest_score_orig, d_lowest_count_orig, d_lowest_indices_orig,  // ADDED COUNT AND INDICES
            d_last_score_orig
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double avg_time = (t1 - t0) / (double)loope;

    // copy back pair-level arrays & aggregated per-original arrays
    long long pairs = total_pair_chunks;
    int* h_pair_distances = (int*)malloc((size_t)pairs * sizeof(int));
    int* h_pair_zcounts = (int*)malloc((size_t)pairs * sizeof(int));
    int* h_pair_zindices = (int*)malloc((size_t)pairs * MAX_HITS * sizeof(int));
    if (!h_pair_distances || !h_pair_zcounts || !h_pair_zindices) {
        fprintf(stderr, "OOM host pair buffers\n");
        return NULL;
    }

    CUDA_CHECK(cudaMemcpy(h_pair_distances, d_pair_distances, (size_t)pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zcounts, d_pair_zcounts, (size_t)pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zindices, d_pair_zindices, (size_t)pairs * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_lowest_score_orig, d_lowest_score_orig, 
    (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_count_orig, d_lowest_count_orig, 
        (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_indices_orig, d_lowest_indices_orig, 
        (size_t)num_queries * num_orig_refs * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_last_score_orig, d_last_score_orig, 
        (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));

    // Correct last scores on host
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long orig_pair_idx = (long long)q * num_orig_refs + orig;
            int orig_len = orig_ref_lens[orig];
            int correct_last_score = 0x7f7f7f7f; // Start with sentinel

            // Find the chunk that actually ends at the original reference length
            for (int ci_idx = 0; ci_idx < orig_chunk_counts[orig]; ++ci_idx) {
                int chunk_id = orig_chunk_lists[orig][ci_idx];
                int chunk_start = part_refs.chunk_starts[chunk_id];
                int chunk_len = part_refs.chunk_lens[chunk_id];

                // This chunk contains the true end if: chunk_start + chunk_len == orig_len
                if (chunk_start + chunk_len == orig_len) {
                    long long pair_idx = (long long)q * num_chunks + chunk_id;
                    correct_last_score = h_pair_distances[pair_idx];
                    break;
                }
            }

            // If we found a valid last score, override the GPU-computed one
            if (correct_last_score != 0x7f7f7f7f) {
                h_last_score_orig[orig_pair_idx] = correct_last_score;
            }
        }
    }

    // ALLOCATE OUTPUT ARRAYS
    long long total_pairs = (long long)num_queries * num_orig_refs;
    
    gpu_args->output_lowest_scores = (int*)malloc(total_pairs * sizeof(int));
    gpu_args->output_lowest_counts = (int*)malloc(total_pairs * sizeof(int));
    gpu_args->output_last_scores = (int*)malloc(total_pairs * sizeof(int));
    gpu_args->output_hit_counts = (int*)malloc(total_pairs * sizeof(int));
    gpu_args->output_lowest_indices = (int**)malloc(total_pairs * sizeof(int*));
    gpu_args->output_hit_indices = (int**)malloc(total_pairs * sizeof(int*));
    
    // Copy simple arrays
    memcpy(gpu_args->output_lowest_scores, h_lowest_score_orig, total_pairs * sizeof(int));
    memcpy(gpu_args->output_lowest_counts, h_lowest_count_orig, total_pairs * sizeof(int));
    memcpy(gpu_args->output_last_scores, h_last_score_orig, total_pairs * sizeof(int));


    // Aggregate hits per original ref on host (dedupe & sort)
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long idx = (long long)q * num_orig_refs + orig;
            
            // -------- Aggregate hit indices (exact matches) --------
            size_t cap = 4096;
            int *acc = (int*)malloc(sizeof(int) * cap);
            size_t acc_n = 0;
            
            for (int ci_idx = 0; ci_idx < orig_chunk_counts[orig]; ++ci_idx) {
                int chunk_id = orig_chunk_lists[orig][ci_idx];
                long long pair_idx = (long long)q * num_chunks + chunk_id;
                int zc = h_pair_zcounts[pair_idx];
                
                for (int k = 0; k < zc && k < MAX_HITS; ++k) {
                    int val = h_pair_zindices[pair_idx * MAX_HITS + k];
                    if (val >= 0) {
                        if (acc_n >= cap) { 
                            cap *= 2; 
                            acc = (int*)realloc(acc, sizeof(int) * cap); 
                        }
                        acc[acc_n++] = val;
                    }
                }
            }
            
            // Sort and deduplicate hits
            if (acc_n > 0) {
                qsort(acc, acc_n, sizeof(int), compare_ints);
                size_t write = 1;
                for (size_t i = 1; i < acc_n; ++i) 
                    if (acc[i] != acc[write-1]) acc[write++] = acc[i];
                acc_n = write;
            }
            
            // Store hits in output
            gpu_args->output_hit_counts[idx] = acc_n;
            if (acc_n > 0) {
                gpu_args->output_hit_indices[idx] = (int*)malloc(acc_n * sizeof(int));
                memcpy(gpu_args->output_hit_indices[idx], acc, acc_n * sizeof(int));
            } else {
                gpu_args->output_hit_indices[idx] = NULL;
            }
            
            // -------- Process lowest score indices --------
            int lowest_cnt = h_lowest_count_orig[idx];
            
            if (lowest_cnt > 0) {
                long long indices_base = idx * MAX_HITS;
                int* lowest_arr = (int*)malloc(sizeof(int) * MIN(lowest_cnt, MAX_HITS));
                int valid_count = 0;
                
                for (int k = 0; k < MIN(lowest_cnt, MAX_HITS); ++k) {
                    int idx_val = h_lowest_indices_orig[indices_base + k];
                    if (idx_val >= 0) {
                        lowest_arr[valid_count++] = idx_val;
                    }
                }
                
                // Sort and deduplicate lowest indices
                if (valid_count > 0) {
                    qsort(lowest_arr, valid_count, sizeof(int), compare_ints);
                    int write = 1;
                    for (int i = 1; i < valid_count; ++i) {
                        if (lowest_arr[i] != lowest_arr[write-1]) {
                            lowest_arr[write++] = lowest_arr[i];
                        }
                    }
                    valid_count = write;
                    
                    // Store in output
                    gpu_args->output_lowest_indices[idx] = (int*)malloc(valid_count * sizeof(int));
                    memcpy(gpu_args->output_lowest_indices[idx], lowest_arr, valid_count * sizeof(int));
                    gpu_args->output_lowest_counts[idx] = valid_count; // Update with actual count
                } else {
                    gpu_args->output_lowest_indices[idx] = NULL;
                    gpu_args->output_lowest_counts[idx] = 0;
                }
                
                free(lowest_arr);
            } else {
                gpu_args->output_lowest_indices[idx] = NULL;
            }
            
            free(acc);
        }
    }

    // Store execution time
    gpu_args->avg_execution_time = avg_time;
    gpu_args->success = 1;
    
    // NOW print results (using the stored output)
    printf("\n=== GPU Results ===\n");
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            long long idx = (long long)q * num_orig_refs + orig;
            
            printf("----------------------------------------------------------------------------\n");
            printf("Pair: Q%d(%d) Vs R%d(%d)\n", q+1, q_lens[q], orig+1, orig_ref_lens[orig]);
            printf("Number of Hits: %d\n", gpu_args->output_hit_counts[idx]);
            
            if (gpu_args->output_hit_counts[idx] > 0) {
                printf("Hit Indexes: [");
                for (int i = 0; i < gpu_args->output_hit_counts[idx]; ++i) {
                    if (i) printf(",");
                    printf("%d", gpu_args->output_hit_indices[idx][i]);
                }
                printf("]\n");
            } else {
                printf("Hit Indexes: N/A\n");
            }
            
            int lowest_score = gpu_args->output_lowest_scores[idx];
            int lowest_cnt = gpu_args->output_lowest_counts[idx];
            
            if (lowest_score == 0x7f7f7f7f) {
                printf("Lowest Score: N/A\n");
                printf("Lowest Score Indexes: N/A\n");
            } else {
                printf("Lowest Score: %d\n", lowest_score);
                
                if (lowest_cnt > 0 && gpu_args->output_lowest_indices[idx] != NULL) {
                    printf("Lowest Score Indexes: [");
                    for (int i = 0; i < lowest_cnt; ++i) {
                        if (i) printf(",");
                        printf("%d", gpu_args->output_lowest_indices[idx][i]);
                    }
                    printf("]\n");
                } else {
                    printf("Lowest Score Indexes: N/A\n");
                }
            }
            
            int last_score = gpu_args->output_last_scores[idx];
            if (last_score == 0x7f7f7f7f) {
                printf("Last Score: N/A\n");
            } else {
                printf("Last Score: %d\n", last_score);
            }
            
            printf("----------------------------------------------------------------------------\n");
        }
    }
    printf("\n%d loop Average time: %.6f sec.\n", loope, avg_time);

    // cleanup
    CUDA_CHECK(cudaFree(d_Eq_queries)); CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_refs));
    CUDA_CHECK(cudaFree(d_q_lens)); CUDA_CHECK(cudaFree(d_ref_lens));
    CUDA_CHECK(cudaFree(d_chunk_starts)); CUDA_CHECK(cudaFree(d_chunk_to_orig)); CUDA_CHECK(cudaFree(d_orig_ref_lens));
    CUDA_CHECK(cudaFree(d_pair_distances)); CUDA_CHECK(cudaFree(d_pair_zcounts)); CUDA_CHECK(cudaFree(d_pair_zindices));
    CUDA_CHECK(cudaFree(d_lowest_score_orig));
    CUDA_CHECK(cudaFree(d_lowest_count_orig));
    CUDA_CHECK(cudaFree(d_lowest_indices_orig));
    CUDA_CHECK(cudaFree(d_last_score_orig));
    
    free(h_Eq_queries); free(h_queries); free(h_refs);
    free(h_pair_distances); free(h_pair_zcounts); free(h_pair_zindices);

    free(h_lowest_score_orig);
    free(h_lowest_count_orig);
    free(h_lowest_indices_orig);
    free(h_last_score_orig);    free(h_ref_lens); free(q_lens);

    free_orig_to_chunk_mapping(orig_chunk_counts, orig_chunk_lists, num_orig_refs);
    free_partitioned_refs(&part_refs);

    free(orig_ref_lens);

    return NULL;
}


// ---------- Host main ----------

int main() {
    printf("=== Hyyrö Bit-Vector Levenshtein with Partitioning ===\n");
    printf("Loading queries and references from memory using loading.h\n");

    int num_queries = 0;
    int num_orig_refs = 0;

    // Load queries and references
    char** query_seqs = load_queries(&num_queries);
    char **gpu_refs = NULL;
    int *gpu_ref_lens = NULL;
    char** orig_refs = load_references_gpu_fpga(&num_orig_refs, &gpu_refs, &gpu_ref_lens, strlen(query_seqs[0]));

    printf("Loaded %d queries and %d references\n", num_queries, num_orig_refs);

    // Prepare arguments for GPU thread
    GPUArgs gpu_args = {
        .num_queries = num_queries,
        .num_orig_refs = num_orig_refs,
        .query_seqs = query_seqs,
        .orig_refs = gpu_refs,
        .output_lowest_scores = NULL,
        .output_lowest_indices = NULL,
        .output_lowest_counts = NULL,
        .output_last_scores = NULL,
        .output_hit_indices = NULL,
        .output_hit_counts = NULL,
        .avg_execution_time = 0.0,
        .success = 0
    };

    // Create GPU thread
    pthread_t gpu_thread;
    pthread_create(&gpu_thread, NULL, run_hyyro_gpu, &gpu_args);

    // ========== Run FPGA for ALL query-ref pairs (in main thread) ==========
    printf("\n=== Starting FPGA Processing ===\n");
    FPGAResult** fpga_results = send_and_run_fpga_multi(num_queries, num_orig_refs);
    printf("=== FPGA Processing Complete ===\n\n");

    // Wait for GPU thread to finish
    pthread_join(gpu_thread, NULL);

    // Check GPU success
    if (!gpu_args.success) {
        printf("GPU computation failed!\n");
        free_fpga_results_multi(fpga_results, num_queries, num_orig_refs);
        return -1;
    }

    // ========== MERGE RESULTS FOR ALL QUERY-REF PAIRS ==========
    printf("\n\n===== Merged Results for All Pairs =====\n\n");

    // Allocate arrays to store merged results for all pairs
    int total_pairs = num_queries * num_orig_refs;
    int* merged_lowest_scores = (int*)malloc(total_pairs * sizeof(int));
    int** merged_lowest_indices = (int**)malloc(total_pairs * sizeof(int*));
    int* merged_index_counts = (int*)malloc(total_pairs * sizeof(int));

    for (int q = 0; q < num_queries; q++) {
        for (int r = 0; r < num_orig_refs; r++) {
            long long idx = (long long)q * num_orig_refs + r;
            
            printf("=============================================================\n");
            printf("Merging: Query %d vs Reference %d\n", q, r);
            printf("=============================================================\n");

            // Get GPU results for this pair
            int lowest_gpu = gpu_args.output_lowest_scores[idx];
            int gpu_lowest_count = gpu_args.output_lowest_counts[idx];
            
            // Get FPGA results for this pair
            int lowest_fpga = fpga_results[q][r].lowest_score;
            int fpga_lowest_count = fpga_results[q][r].num_lowest_indexes;

            printf("GPU:  Lowest Score = %d, Count = %d\n", lowest_gpu, gpu_lowest_count);
            printf("FPGA: Lowest Score = %d, Count = %d\n", lowest_fpga, fpga_lowest_count);

            // Determine merged lowest score
            int final_merged_lowest_score = -1;
            if (lowest_gpu != 0x7f7f7f7f && lowest_fpga != -1) {
                final_merged_lowest_score = MIN(lowest_gpu, lowest_fpga);
            } else if (lowest_gpu != 0x7f7f7f7f) {
                final_merged_lowest_score = lowest_gpu;
            } else {
                final_merged_lowest_score = lowest_fpga;
            }

            merged_lowest_scores[idx] = final_merged_lowest_score;

            // Count total indexes
            int total_index_count = 0;
            if (final_merged_lowest_score != -1) {
                if (lowest_gpu == final_merged_lowest_score) {
                    total_index_count += gpu_lowest_count;
                }
                if (lowest_fpga == final_merged_lowest_score) {
                    total_index_count += fpga_lowest_count;
                }
            }

            // Allocate and fill merged index array
            if (total_index_count > 0) {
                merged_lowest_indices[idx] = (int*)malloc(total_index_count * sizeof(int));
                int write_pos = 0;

                // Add GPU indexes if they match
                if (lowest_gpu == final_merged_lowest_score && gpu_args.output_lowest_indices[idx] != NULL) {
                    for (int i = 0; i < gpu_lowest_count; i++) {
                        merged_lowest_indices[idx][write_pos++] = gpu_args.output_lowest_indices[idx][i];
                    }
                }

                // Add FPGA indexes if they match (with offset adjustment)
                if (lowest_fpga == final_merged_lowest_score && fpga_results[q][r].lowest_indexes != NULL) {
                    int gpu_len = (int)(strlen(orig_refs[r]) * GPU_SPEED_RATIO);
                    int query_len = (int)strlen(query_seqs[q]);  // Get actual query length
                    for (int i = 0; i < fpga_lowest_count; i++) {
                        // Adjust: fpga_index + gpu_length - query_length
                        merged_lowest_indices[idx][write_pos++] = fpga_results[q][r].lowest_indexes[i] + gpu_len - query_len + 1;
                    }
                }

                merged_index_counts[idx] = write_pos;

                // Sort merged indexes
                qsort(merged_lowest_indices[idx], merged_index_counts[idx], sizeof(int), compare_ints);
            } else {
                merged_lowest_indices[idx] = NULL;
                merged_index_counts[idx] = 0;
            }

            // Print merged results for this pair
            printf("\nMerged Results:\n");
            printf("  Final Lowest Score: %d\n", merged_lowest_scores[idx]);
            printf("  Total Index Count: %d\n", merged_index_counts[idx]);
            if (merged_index_counts[idx] > 0) {
                printf("  Merged Lowest Indexes: [");
                for (int i = 0; i < merged_index_counts[idx]; i++) {
                    if (i > 0) printf(", ");
                    printf("%d", merged_lowest_indices[idx][i]);
                }
                printf("]\n");
            } else {
                printf("  Merged Lowest Indexes: N/A\n");
            }
            printf("\n");
        }
    }

    printf("\n=============================================================\n");
    printf("Summary: All %d query-reference pairs processed and merged\n", total_pairs);
    printf("=============================================================\n");

    // ========== NOW USE THE MERGED RESULTS ==========
    // Example: Access specific pair
    int example_q = 0;
    int example_r = 0;
    long long example_idx = (long long)example_q * num_orig_refs + example_r;
    
    printf("\nExample Access - Q%d vs R%d:\n", example_q, example_r);
    printf("  Merged Lowest Score: %d\n", merged_lowest_scores[example_idx]);
    printf("  Merged Index Count: %d\n", merged_index_counts[example_idx]);

    // ========== CLEANUP ==========
    
    // Free FPGA results
    free_fpga_results_multi(fpga_results, num_queries, num_orig_refs);

    // Free merged results
    for (int i = 0; i < total_pairs; i++) {
        if (merged_lowest_indices[i]) free(merged_lowest_indices[i]);
    }
    free(merged_lowest_scores);
    free(merged_lowest_indices);
    free(merged_index_counts);

    // Cleanup GPU results
    for (long long i = 0; i < total_pairs; i++) {
        if (gpu_args.output_hit_indices[i]) free(gpu_args.output_hit_indices[i]);
        if (gpu_args.output_lowest_indices[i]) free(gpu_args.output_lowest_indices[i]);
    }

    free(gpu_args.output_lowest_scores);
    free(gpu_args.output_lowest_indices);
    free(gpu_args.output_lowest_counts);
    free(gpu_args.output_last_scores);
    free(gpu_args.output_hit_indices);
    free(gpu_args.output_hit_counts);

    // Cleanup queries/references
    for (int i = 0; i < num_queries; i++) free(query_seqs[i]);
    free(query_seqs);
    for (int i = 0; i < num_orig_refs; i++) free(orig_refs[i]);
    free(orig_refs);

    return 0;
}