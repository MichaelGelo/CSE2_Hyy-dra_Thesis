// finalcuda.cu


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

    // Load shared Eq table
    for (int i = tid; i < 256; i += blockDim.x) {
        s_Eq[i] = d_Eq_queries[(long long)q * 256LL + i];
    }
    __syncthreads();

    int qlen = d_q_lens[q];
    
    // ==================== PHASE 1: Compute distances and update lowest scores ONLY ====================
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

        // Write pair-level results (hits at score=0)
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

        // PHASE 1: Only update the lowest score (no indices yet!)
        long long orig_pair_idx = (long long)q * num_orig_refs + orig;
        if (local_lowest_count > 0) {
            atomicMin(&d_lowest_score_orig[orig_pair_idx], local_lowest_score);
        }
    }
    
    // ==================== CRITICAL SYNCHRONIZATION ====================
    // Wait for ALL threads in this block to finish updating scores
    __syncthreads();
    __threadfence();  // Ensure writes are visible globally
    
    // ==================== PHASE 2: Collect indices matching FINAL best scores ====================
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        const char* refptr = &d_refs[(size_t)c * MAX_LENGTH];
        int rlen = d_ref_lens[c];
        int chunk_start = d_chunk_starts[c];
        int orig = d_chunk_to_orig[c];

        int local_zero_indices[MAX_HITS];
        int local_zero_count = 0;
        int local_lowest_score = INT_MAX;
        int local_lowest_indices[MAX_HITS];
        int local_lowest_count = 0;

        int dist = bit_vector_levenshtein_local(qlen, refptr, rlen, s_Eq,
            local_zero_indices, &local_zero_count, 
            &local_lowest_score, local_lowest_indices, &local_lowest_count);

        long long orig_pair_idx = (long long)q * num_orig_refs + orig;
        long long orig_indices_base = orig_pair_idx * MAX_HITS;
        int final_best_score = d_lowest_score_orig[orig_pair_idx];

        if (local_lowest_score == final_best_score && local_lowest_count > 0) {
            for (int k = 0; k < local_lowest_count; ++k) {
                int global_lowest_pos = chunk_start + local_lowest_indices[k];
                
                // Atomically reserve a slot for this index
                int slot = atomicAdd(&d_lowest_count_orig[orig_pair_idx], 1);
                
                if (slot < MAX_HITS) {
                    d_lowest_indices_orig[orig_indices_base + slot] = global_lowest_pos;
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

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    float total_ms = 0.0f;
    for (int it = 0; it < loope; ++it) {
        CUDA_CHECK(cudaEventRecord(ev_start, 0));
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
        CUDA_CHECK(cudaEventRecord(ev_end, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
    }
    float iter_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_start, ev_end));
    total_ms += iter_ms;
    CUDA_CHECK(cudaDeviceSynchronize());

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
    gpu_args->avg_execution_time = total_ms / (loope*num_queries) / 1000.0; // seconds;
    gpu_args->success = 1;


    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    
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
    printf("\n%d loop Average time: %.6f sec.\n", loope, (total_ms / (loope*num_queries) / 1000.0));


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

// ---------- MAIN FUNCTION ----------

int main() {
    printf("=== Hyyrö Bit-Vector Levenshtein with Adaptive Partitioning ===\n");
    printf("Loading queries and references from memory using loading.h\n");

    int num_queries = 0, num_orig_refs = 0;
    char **query_seqs = load_queries(&num_queries);
    if (!query_seqs || num_queries == 0) { fprintf(stderr, "Failed to load queries\n"); return -1; }

    char **orig_refs = parse_fasta_file(REFERENCE_FILE, &num_orig_refs);
    if (!orig_refs || num_orig_refs <= 0) { fprintf(stderr, "Failed to load references\n"); return -1; }

    printf("Loaded %d queries and %d references\n", num_queries, num_orig_refs);

    //SpeedRatioState ratio_state = {0.0, 0.0, GPU_SPEED_RATIO, FPGA_SPEED_RATIO, 0};
    int query_len = (int)strlen(query_seqs[0]);

    char **all_gpu_refs = (char**)malloc(sizeof(char*) * num_orig_refs);
    int *all_gpu_ref_lens = (int*)malloc(sizeof(int) * num_orig_refs);
    if (!all_gpu_refs || !all_gpu_ref_lens) { fprintf(stderr, "OOM\n"); return -1; }

    GPUArgs *all_gpu_results = (GPUArgs*)calloc(num_orig_refs, sizeof(GPUArgs));
    FPGAResult **all_fpga_results = (FPGAResult**)calloc(num_orig_refs, sizeof(FPGAResult*));

    // Precompute ratio thread handle and args
    pthread_t ratio_thread;
    RatioThreadArgs ratio_args = {0};

    float gpu_ratio = GPU_SPEED_RATIO;   // initial
    float fpga_ratio = FPGA_SPEED_RATIO;

    double total_gpu_time = 0.0;
    double total_fpga_time = 0.0;

    // allocate timeline arrays
    float* ratio_used_gpu = (float*)malloc(num_orig_refs * sizeof(float));
    float* ratio_used_fpga = (float*)malloc(num_orig_refs * sizeof(float));
    float* ratio_updated_gpu = (float*)malloc(num_orig_refs * sizeof(float));
    float* ratio_updated_fpga = (float*)malloc(num_orig_refs * sizeof(float));
    double* gpu_time_per_ref = (double*)malloc(num_orig_refs * sizeof(double));
    double* fpga_time_per_ref = (double*)malloc(num_orig_refs * sizeof(double));

    // check allocations
    if (!ratio_used_gpu || !ratio_used_fpga || !ratio_updated_gpu || !ratio_updated_fpga ||
        !gpu_time_per_ref || !fpga_time_per_ref) {
        fprintf(stderr, "Allocation failure for timelines\n");
        exit(EXIT_FAILURE);
    }

    // The main per-reference loop (replace your old loop body with this)
    for (int ref_idx = 0; ref_idx < num_orig_refs; ref_idx++) {
        printf("\n============================================================\n");
        printf("Processing Reference %d/%d\n", ref_idx + 1, num_orig_refs);
        printf("Current GPU ratio: %.3f, FPGA ratio: %.3f\n", gpu_ratio, fpga_ratio);

        // store the ratio that will be USED for splitting this reference
        ratio_used_gpu[ref_idx]  = gpu_ratio;
        ratio_used_fpga[ref_idx] = fpga_ratio;

        // Split reference according to gpu_ratio
        split_reference_for_fpga_gpu(orig_refs[ref_idx], query_len,
                                    &all_gpu_refs[ref_idx],
                                    &all_gpu_ref_lens[ref_idx],
                                    ref_idx,
                                    gpu_ratio);

        // Start GPU thread (unchanged)
        GPUArgs gpu_args = {
            .num_queries = num_queries,
            .num_orig_refs = 1,
            .query_seqs = query_seqs,
            .orig_refs = &all_gpu_refs[ref_idx],
            .output_lowest_scores = NULL,
            .output_lowest_indices = NULL,
            .output_lowest_counts = NULL,
            .output_last_scores = NULL,
            .output_hit_indices = NULL,
            .output_hit_counts = NULL,
            .avg_execution_time = 0.0,
            .success = 0
        };

        pthread_t gpu_thread;
        if (pthread_create(&gpu_thread, NULL, run_hyyro_gpu, &gpu_args) != 0) {
            fprintf(stderr, "ERROR: failed to create GPU thread for ref %d\n", ref_idx);
            exit(EXIT_FAILURE);
        }

// ====================================================================
        // TCP RAM-Stream FPGA Execution
        // ====================================================================
        double fpga_wall_start = now_seconds();
        
        // 1. Calculate pointer to FPGA's portion of data
        int total_len = strlen(orig_refs[ref_idx]);
        int gpu_processed_len = (int)(total_len * gpu_ratio);
        int overlap = query_len - 1;
        int fpga_start_idx = gpu_processed_len - overlap;
        if (fpga_start_idx < 0) fpga_start_idx = 0;

        const char* fpga_ref_ptr = &orig_refs[ref_idx][fpga_start_idx];

        // 2. Allocate Results Array
        FPGAResult* fpga_results = (FPGAResult*)malloc(num_queries * sizeof(FPGAResult));
        if (!fpga_results) { fprintf(stderr, "Malloc failed for FPGA results\n"); exit(1); }

        // 3. Process Queries & Accumulate PURE Hardware Time
        double fpga_pure_hw_ms = 0.0; // <--- NEW ACCUMULATOR

        for (int q = 0; q < num_queries; q++) {
            // Call the function (works for both Loop or Batch mode)
            fpga_results[q] = run_fpga_tcp_ram(fpga_ref_ptr, query_seqs[q]);
            
            // EXTRACT PURE TIME: This comes from the Python server's response
            // "Hardware Time: X ms"
            fpga_pure_hw_ms = fpga_results[q].hw_exec_time_ms; //bobo
            
            // Adjust indices
            if (fpga_results[q].num_lowest_indexes > 0) {
                for(int k=0; k<fpga_results[q].num_lowest_indexes; k++) {
                    fpga_results[q].lowest_indexes[k] += fpga_start_idx;
                }
            }
        }

        // 4. Calculate Final Times
        double fpga_total_wall_sec = now_seconds() - fpga_wall_start; // Includes Network (Use for Load Balancing)
        double fpga_pure_hw_sec = fpga_pure_hw_ms / 1000.0;           // Excludes Network (Use for Reporting)

        // ====================================================================

        // Wait for GPU
        pthread_join(gpu_thread, NULL);
        double gpu_time = gpu_args.avg_execution_time; // This is ALREADY Pure Kernel Time

        // === PRINT COMPARISON (Now Fair!) ===
        printf("=== GPU Kernel Time:     %.6f sec ===\n", gpu_time);
        printf("=== FPGA Hardware Time:  %.6f sec === (Total w/ Network: %.6f)\n", fpga_pure_hw_sec, fpga_total_wall_sec);

        // Store PURE time for the final results summary
        gpu_time_per_ref[ref_idx] = gpu_time;
        fpga_time_per_ref[ref_idx] = fpga_pure_hw_sec; 
        total_gpu_time += gpu_time;
        total_fpga_time += fpga_pure_hw_sec;

        // CRITICAL: Load Balancer must use TOTAL Wall Time
        // The network bottleneck is real; if we hide it from the balancer, 
        // it will send too much data and crash the system.
        RatioThreadArgs ratio_args;
        ratio_args.prev_gpu_time = gpu_time;
        ratio_args.prev_fpga_time = fpga_pure_hw_sec; // P THIS AS TOTAL (bobo ka)
        ratio_args.old_gpu_ratio  = gpu_ratio;
        ratio_args.taken = 0;
        ratio_args.new_gpu_ratio = gpu_ratio; 

        pthread_t ratio_thread;
        if (pthread_create(&ratio_thread, NULL, ratio_thread_func, &ratio_args) != 0) {
            fprintf(stderr, "ERROR: failed to create ratio thread\n");
            exit(EXIT_FAILURE);
        }

        // Wait for handshake
        pthread_mutex_lock(&ratio_mutex);
        while (ratio_args.taken == 0) {
            pthread_cond_wait(&ratio_cond, &ratio_mutex);
        }
        pthread_mutex_unlock(&ratio_mutex);

        // Join ratio thread
        pthread_join(ratio_thread, NULL);

        // Update ratios
        gpu_ratio = ratio_args.new_gpu_ratio;
        fpga_ratio = 1.0f - gpu_ratio;

        // Store updated ratio
        ratio_updated_gpu[ref_idx]  = gpu_ratio;
        ratio_updated_fpga[ref_idx] = fpga_ratio;

        // Store results
        all_gpu_results[ref_idx]  = gpu_args;
        all_fpga_results[ref_idx] = fpga_results;
    }

    printf("\n\n===== HETEROGENEOUS RESULTS =====\n\n");

    int total_pairs = num_queries * num_orig_refs;
    int* merged_lowest_scores = (int*)malloc(total_pairs * sizeof(int));
    int** merged_lowest_indices = (int**)malloc(total_pairs * sizeof(int*));
    int* merged_index_counts = (int*)malloc(total_pairs * sizeof(int));

    for (int r = 0; r < num_orig_refs; r++){
        for (int q = 0; q < num_queries; q++){
            long long idx = (long long)q * num_orig_refs + r;

            printf("=============================================================\n");
            printf("Query %d vs Reference %d\n", q+1, r+1);
            printf("=============================================================\n");

            long long gpu_idx = q;

            int lowest_gpu = all_gpu_results[r].output_lowest_scores[gpu_idx];
            int gpu_lowest_count = all_gpu_results[r].output_lowest_counts[gpu_idx];

            int lowest_fpga = all_fpga_results[r][q].lowest_score;
            int fpga_lowest_count = all_fpga_results[r][q].num_lowest_indexes;

            int last_score = all_fpga_results[r][q].final_score;

            printf("GPU:  Lowest Score = %d, Count = %d\n", lowest_gpu, gpu_lowest_count);
            printf("FPGA: Lowest Score = %d, Count = %d\n", lowest_fpga, fpga_lowest_count);

            int final_merged_lowest_score = -1;
            if (lowest_gpu != 0x7f7f7f7f && lowest_fpga != -1) {
                final_merged_lowest_score = MIN(lowest_gpu, lowest_fpga);
            } else if (lowest_gpu != 0x7f7f7f7f) {
                final_merged_lowest_score = lowest_gpu;
            } else {
                final_merged_lowest_score = lowest_fpga;
            }

            merged_lowest_scores[idx] = final_merged_lowest_score;

            int total_index_count = 0;
            if (final_merged_lowest_score != -1) {
                if (lowest_gpu == final_merged_lowest_score) {
                    total_index_count += gpu_lowest_count;
                }
                if (lowest_fpga == final_merged_lowest_score) {
                    total_index_count += fpga_lowest_count;
                }
            }

            merged_index_counts[idx] = total_index_count;

            if (total_index_count > 0) {
                merged_lowest_indices[idx] = (int*)malloc(total_index_count * sizeof(int));
                int write_pos = 0;

                if (lowest_gpu == final_merged_lowest_score &&
                    all_gpu_results[r].output_lowest_indices[gpu_idx] != NULL) {
                    for (int i = 0; i < gpu_lowest_count; i++) {
                        merged_lowest_indices[idx][write_pos++] =
                            all_gpu_results[r].output_lowest_indices[gpu_idx][i];
                    }
                }

                if (lowest_fpga == final_merged_lowest_score &&
                    all_fpga_results[r][q].lowest_indexes != NULL) {
                    int gpu_len = all_gpu_ref_lens[r];
                    for (int i = 0; i < fpga_lowest_count; i++) {
                        merged_lowest_indices[idx][write_pos++] =
                            all_fpga_results[r][q].lowest_indexes[i] +
                            gpu_len - query_len + 1;
                    }
                }

                qsort(merged_lowest_indices[idx], merged_index_counts[idx],
                    sizeof(int), compare_ints);
            } else {
                merged_lowest_indices[idx] = NULL;
            }

            printf("\nResults:\n");
            printf("  Lowest Score: %d\n", merged_lowest_scores[idx]);

            int hit_index;
            if (merged_lowest_scores[idx] > 0) hit_index = 0;
            else hit_index = merged_index_counts[idx];

            printf("  Number of Hits: %d\n", hit_index);
            printf("  Total Index Count: %d\n", merged_index_counts[idx]);

            if (merged_index_counts[idx] > 0) {
                printf("  Lowest Indexes: [");
                for (int i = 0; i < merged_index_counts[idx]; i++) {
                    if (i > 0) printf(", ");
                    printf("%d", merged_lowest_indices[idx][i]);
                }
                printf("]\n");
            } else {
                printf("  Hit Indexes: [ ]\n");
            }

            printf("  Last Score: %d\n", last_score);
            printf("\n");
        }
    }

    // Summary: print totals and timeline
    printf("\n=============================================================\n");
    printf("Summary: All %d pairs processed with adaptive ratios\n", num_queries * num_orig_refs);

    printf("Total GPU time: %.6f sec\n", total_gpu_time);
    printf("Total FPGA time: %.6f sec\n", total_fpga_time);

    printf("\nRatios Per Reference:\n");
    for (int i = 0; i < num_orig_refs; i++) {
        printf("Ref%d used:    GPU %.3f FPGA %.3f\n", i+1, ratio_used_gpu[i], ratio_used_fpga[i]);
        printf("Ref%d times:   GPU %.6f sec, FPGA %.6f sec\n\n", i+1, gpu_time_per_ref[i], fpga_time_per_ref[i]);
    }
    printf("Final GPU ratio: %.3f, FPGA ratio: %.3f\n", gpu_ratio, fpga_ratio);
    printf("=============================================================\n");

    // cleanup timeline arrays
    free(ratio_used_gpu);
    free(ratio_used_fpga);
    free(ratio_updated_gpu);
    free(ratio_updated_fpga);
    free(gpu_time_per_ref);
    free(fpga_time_per_ref);

    return 0;
}