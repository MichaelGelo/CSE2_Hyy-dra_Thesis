// unified_leven_partitioned_gpu.cu
// Combined partition + Hyyrö bit-vector Levenshtein with GPU-side aggregation
// Compile:
//   nvcc -O3 unified_leven_partitioned_gpu.cu C_utils.c -o unified_leven_partitioned_gpu
//
// Run:
//   ./unified_leven_partitioned_gpu
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
#define query_file "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/Queries/que4_256.fasta"
#define reference_file "/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources/References/ref5_50M.fasta"
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
    uint64_t carry = 0;
    uint64_t sum = a->w[0] + b->w[0];
    carry = (sum < a->w[0]);
    out->w[0] = sum;
    sum = a->w[1] + b->w[1] + carry;
    carry = (sum < a->w[1]) || (carry && sum == a->w[1]);
    out->w[1] = sum;
    sum = a->w[2] + b->w[2] + carry;
    carry = (sum < a->w[2]) || (carry && sum == a->w[2]);
    out->w[2] = sum;
    sum = a->w[3] + b->w[3] + carry;
    carry = (sum < a->w[3]) || (carry && sum == a->w[3]);
    out->w[3] = sum;
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
    int* lowest_index_local)   // returns local j for lowest
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

        int dist = bit_vector_levenshtein_local(qlen, refptr, rlen, s_Eq, local_zero_indices, &local_zero_count, &local_lowest_val, &local_lowest_idx);

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

        // If there was a zero (score==0) we still need to ensure lowest is 0 and index recorded
        // Note: local_lowest_val will be 0 in that case and handled above.

        // 2) set last score if this chunk contains the final base of the original reference
        // Condition: chunk_start + (rlen - 1) == orig_ref_len - 1
        int orig_len = d_orig_ref_lens[orig];
        if ((long long)chunk_start + (long long)rlen == (long long)orig_len) {
            // this chunk ends exactly at original's final position; write last score
            // We use atomicExch because only the true last chunk should hit this condition,
            // but atomicExch is safe even if multiple threads attempt (they would write same value)
            atomicExch(&d_last_score_orig[orig_pair_idx], dist);
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
    int estimated_chunks = 0;
    for (int i = 0; i < num_orig_refs; ++i) {
        if (orig_ref_lens[i] > PARTITION_THRESHOLD) {
            int nch = (orig_ref_lens[i] + CHUNK_SIZE - 1) / CHUNK_SIZE;
            estimated_chunks += nch;
        } else estimated_chunks += 1;
    }

    char** chunk_seqs = (char**)malloc(sizeof(char*) * estimated_chunks);
    int* chunk_lens = (int*)malloc(sizeof(int) * estimated_chunks);
    int* chunk_starts = (int*)malloc(sizeof(int) * estimated_chunks); // global start in original
    int* chunk_to_orig = (int*)malloc(sizeof(int) * estimated_chunks);

    int chunk_idx = 0;
    for (int r = 0; r < num_orig_refs; ++r) {
        int rlen = orig_ref_lens[r];
        if (rlen > PARTITION_THRESHOLD) {
            int nch = (rlen + CHUNK_SIZE - 1) / CHUNK_SIZE;
            for (int c = 0; c < nch; ++c) {
                int start = c * CHUNK_SIZE;
                int len = CHUNK_SIZE;
                if (start + len > rlen) len = rlen - start;
                int ext_len = len + (qlen0 - 1);
                if (start + ext_len > rlen) ext_len = rlen - start;
                char* s = (char*)malloc(ext_len + 1);
                memcpy(s, orig_refs[r] + start, ext_len);
                s[ext_len] = '\0';
                chunk_seqs[chunk_idx] = s;
                chunk_lens[chunk_idx] = ext_len;
                chunk_starts[chunk_idx] = start;
                chunk_to_orig[chunk_idx] = r;
                chunk_idx++;
            }
        } else {
            int ext_len = rlen;
            char* s = (char*)malloc(ext_len + 1);
            memcpy(s, orig_refs[r], ext_len);
            s[ext_len] = '\0';
            chunk_seqs[chunk_idx] = s;
            chunk_lens[chunk_idx] = ext_len;
            chunk_starts[chunk_idx] = 0;
            chunk_to_orig[chunk_idx] = r;
            chunk_idx++;
        }
    }
    int num_chunks = chunk_idx;
    int num_references = num_chunks; // kernel references = chunks

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

    // build orig->chunk list counts for host aggregation
    int* orig_chunk_counts = (int*)calloc(num_orig_refs, sizeof(int));
    for (int r = 0; r < num_references; ++r) orig_chunk_counts[chunk_to_orig[r]]++;
    int** orig_chunk_lists = (int**)malloc(sizeof(int*) * num_orig_refs);
    for (int i = 0; i < num_orig_refs; ++i) {
        if (orig_chunk_counts[i] > 0) orig_chunk_lists[i] = (int*)malloc(sizeof(int) * orig_chunk_counts[i]);
        else orig_chunk_lists[i] = NULL;
        orig_chunk_counts[i] = 0;
    }
    for (int r = 0; r < num_references; ++r) {
        int o = chunk_to_orig[r];
        orig_chunk_lists[o][orig_chunk_counts[o]++] = r;
    }

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
    int* d_lowest_score_orig = NULL;    // size num_queries * num_orig_refs
    int* d_lowest_index_orig = NULL;
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
    CUDA_CHECK(cudaMalloc((void**)&d_lowest_index_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_last_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int)));

    // host arrays to copy back aggregated orig results
    int* h_lowest_score_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));
    int* h_lowest_index_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));
    int* h_last_score_orig = (int*)malloc((size_t)num_queries * num_orig_refs * sizeof(int));

    // prepare host arrays
    CUDA_CHECK(cudaMemcpy(d_Eq_queries, h_Eq_queries, (size_t)num_queries * 256 * sizeof(bv_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, h_queries_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_refs, h_refs, h_refs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_lens, q_lens, num_queries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_lens, h_ref_lens, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_starts, chunk_starts, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_to_orig, chunk_to_orig, num_references * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig_ref_lens, orig_ref_lens, num_orig_refs * sizeof(int), cudaMemcpyHostToDevice));

    // init per-original arrays on device
    // lowest_score_orig -> initialize to large sentinel INT_MAX
    CUDA_CHECK(cudaMemset(d_lowest_score_orig, 0x7f, (size_t)num_queries * num_orig_refs * sizeof(int)));
    // lowest_index_orig -> -1 sentinel
    CUDA_CHECK(cudaMemset(d_lowest_index_orig, 0xff, (size_t)num_queries * num_orig_refs * sizeof(int)));
    // last_score_orig -> sentinel (use 0x7f to indicate missing)
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

    printf("*** launching kernel: blocks=%d threads=%d shared=%zu num_chunks=%d num_orig=%d\n", blocks, threads, shared_bytes, num_chunks, num_orig_refs);
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

    // copy back pair-level arrays & aggregated per-original arrays
    long long pairs = total_pair_chunks;
    int* h_pair_distances = (int*)malloc((size_t)pairs * sizeof(int));
    int* h_pair_zcounts = (int*)malloc((size_t)pairs * sizeof(int));
    int* h_pair_zindices = (int*)malloc((size_t)pairs * MAX_HITS * sizeof(int));
    if (!h_pair_distances || !h_pair_zcounts || !h_pair_zindices) { fprintf(stderr, "OOM host pair buffers\n"); return -1; }

    CUDA_CHECK(cudaMemcpy(h_pair_distances, d_pair_distances, (size_t)pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zcounts, d_pair_zcounts, (size_t)pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pair_zindices, d_pair_zindices, (size_t)pairs * MAX_HITS * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_lowest_score_orig, d_lowest_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lowest_index_orig, d_lowest_index_orig, (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_last_score_orig, d_last_score_orig, (size_t)num_queries * num_orig_refs * sizeof(int), cudaMemcpyDeviceToHost));

    // Aggregate hits per original ref on host (dedupe & sort)
    for (int q = 0; q < num_queries; ++q) {
        for (int orig = 0; orig < num_orig_refs; ++orig) {
            // gather hits across all chunks that map to this original ref
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

            // dedupe & sort
            if (acc_n > 0) {
                qsort(acc, acc_n, sizeof(int), compare_ints);
                size_t write = 1;
                for (size_t i = 1; i < acc_n; ++i) if (acc[i] != acc[write-1]) acc[write++] = acc[i];
                acc_n = write;
            }

            // print block exactly as requested
            printf("----------------------------------------------------------------------------\n");
            printf("Pair: Q%d(%d) Vs R%d(%d)\n", q+1, q_lens[q], orig+1, orig_ref_lens[orig]);
            printf("Number of Hits: %zu\n", acc_n);
            if (acc_n > 0) {
                printf("Hit Indexes: [");
                for (size_t i = 0; i < acc_n; ++i) {
                    if (i) printf(",");
                    printf("%d", acc[i]);
                }
                printf("]\n");
            } else {
                printf("Hit Indexes: N/A\n");
            }

            // lowest score and index come from GPU aggregated arrays
            int lowest_gpu = h_lowest_score_orig[(long long)q * num_orig_refs + orig];
            int lowest_idx_gpu = h_lowest_index_orig[(long long)q * num_orig_refs + orig];
            // sentinel check: INT_MAX sentinel used (0x7f7f7f7f)
            if (lowest_gpu == 0x7f7f7f7f) {
                printf("Lowest Score: N/A\n");
                printf("Lowest Score Indexes: N/A\n");
            } else {
                printf("Lowest Score: %d\n", lowest_gpu);
                if (lowest_idx_gpu >= 0) printf("Lowest Score Indexes: [%d]\n", lowest_idx_gpu);
                else printf("Lowest Score Indexes: N/A\n");
            }

            // last score: device wrote it only if chunk contained final pos; sentinel check
            int last_gpu = h_last_score_orig[(long long)q * num_orig_refs + orig];
            if (last_gpu == 0x7f7f7f7f) {
                printf("Last Score: N/A\n");
            } else {
                printf("Last Score: %d\n", last_gpu);
            }

            printf("----------------------------------------------------------------------------\n");

            free(acc);
        }
    }

    printf("%d loop Average time: %.6f sec.\n", loope, avg_time);

    // cleanup
    CUDA_CHECK(cudaFree(d_Eq_queries)); CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_refs));
    CUDA_CHECK(cudaFree(d_q_lens)); CUDA_CHECK(cudaFree(d_ref_lens));
    CUDA_CHECK(cudaFree(d_chunk_starts)); CUDA_CHECK(cudaFree(d_chunk_to_orig)); CUDA_CHECK(cudaFree(d_orig_ref_lens));
    CUDA_CHECK(cudaFree(d_pair_distances)); CUDA_CHECK(cudaFree(d_pair_zcounts)); CUDA_CHECK(cudaFree(d_pair_zindices));
    CUDA_CHECK(cudaFree(d_lowest_score_orig)); CUDA_CHECK(cudaFree(d_lowest_index_orig)); CUDA_CHECK(cudaFree(d_last_score_orig));

    free(h_Eq_queries); free(h_queries); free(h_refs);
    free(h_pair_distances); free(h_pair_zcounts); free(h_pair_zindices);
    free(h_lowest_score_orig); free(h_lowest_index_orig); free(h_last_score_orig);
    free(h_ref_lens); free(q_lens);
    free(orig_chunk_counts);
    for (int i = 0; i < num_orig_refs; ++i) if (orig_chunk_lists[i]) free(orig_chunk_lists[i]);
    free(orig_chunk_lists);

    for (int i = 0; i < num_references; ++i) free(chunk_seqs[i]);
    free(chunk_seqs); free(chunk_lens); free(chunk_starts); free(chunk_to_orig);

    for (int i = 0; i < num_queries; ++i) free(query_seqs[i]); free(query_seqs);
    for (int i = 0; i < num_orig_refs; ++i) free(orig_refs[i]); free(orig_refs);
    free(orig_ref_lens);

    return 0;
}
