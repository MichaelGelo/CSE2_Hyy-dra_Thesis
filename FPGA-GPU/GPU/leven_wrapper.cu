// leven_wrapper.cu
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <cuda_runtime.h>
#include "C_utils.h"

#define ALIGNMENT_API __attribute__((visibility("default")))

#define MAX_LENGTH (1 << 26)
#define MAX_HITS 1024
#define threadsPerBlock 256
#define BV_WORDS 4
#define MIN(a,b) ((a)<(b)?(a):(b))

// Error check macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while (0)

// ---------------- Bitvector helpers ----------------
typedef struct { uint64_t w[BV_WORDS]; } bv_t;

__host__ __device__ static inline void bv_set_all_unrolled(bv_t *out, uint64_t v) {
    out->w[0]=v; out->w[1]=v; out->w[2]=v; out->w[3]=v;
}
__host__ __device__ static inline void bv_clear_unrolled(bv_t *out) {
    out->w[0]=0ULL; out->w[1]=0ULL; out->w[2]=0ULL; out->w[3]=0ULL;
}
__host__ __device__ static inline void bv_copy_unrolled(bv_t *out, const bv_t *in) {
    out->w[0]=in->w[0]; out->w[1]=in->w[1]; out->w[2]=in->w[2]; out->w[3]=in->w[3];
}
__host__ __device__ static inline void bv_or_unrolled(bv_t *out, const bv_t *a, const bv_t *b) {
    out->w[0]=a->w[0]|b->w[0]; out->w[1]=a->w[1]|b->w[1]; out->w[2]=a->w[2]|b->w[2]; out->w[3]=a->w[3]|b->w[3];
}
__host__ __device__ static inline void bv_and_unrolled(bv_t *out, const bv_t *a, const bv_t *b) {
    out->w[0]=a->w[0]&b->w[0]; out->w[1]=a->w[1]&b->w[1]; out->w[2]=a->w[2]&b->w[2]; out->w[3]=a->w[3]&b->w[3];
}
__host__ __device__ static inline void bv_not_unrolled(bv_t *out, const bv_t *a) {
    out->w[0]=~a->w[0]; out->w[1]=~a->w[1]; out->w[2]=~a->w[2]; out->w[3]=~a->w[3];
}
__host__ __device__ static inline void bv_shl1_unrolled(bv_t *out, const bv_t *in) {
    uint64_t c0=in->w[0]>>63, c1=in->w[1]>>63, c2=in->w[2]>>63;
    out->w[0]=(in->w[0]<<1);
    out->w[1]=(in->w[1]<<1)|c0;
    out->w[2]=(in->w[2]<<1)|c1;
    out->w[3]=(in->w[3]<<1)|c2;
}
__host__ __device__ static inline int bv_test_top_unrolled(const bv_t *v, int query_length) {
    int idx=(query_length-1)/64; int bit=(query_length-1)%64;
    return ((v->w[idx]>>bit)&1ULL)?1:0;
}
__host__ __device__ static inline uint64_t bv_add_unrolled(bv_t *out, const bv_t *a, const bv_t *b) {
    uint64_t carry=0ULL;
    for (int i=0;i<BV_WORDS;i++) {
        uint64_t sum=a->w[i]+b->w[i]+carry;
        carry=(sum<a->w[i]||(sum==a->w[i]&&carry));
        out->w[i]=sum;
    }
    return carry;
}
__host__ __device__ static inline void bv_xor_unrolled(bv_t *out, const bv_t *a, const bv_t *b) {
    out->w[0]=a->w[0]^b->w[0]; out->w[1]=a->w[1]^b->w[1]; out->w[2]=a->w[2]^b->w[2]; out->w[3]=a->w[3]^b->w[3];
}

// ---------------- Bit-vector Levenshtein ----------------
__host__ __device__ static inline int bit_vector_levenshtein_local(
    int query_length,
    const char *reference,
    int reference_length,
    const bv_t *Eq,
    int *lowest,
    int *lowest_index_local)
{
    bv_t Pv,Mv,Ph,Mh,Xv,Xh,Xp,tmp1,tmp2,tmp3,tmp4;
    bv_set_all_unrolled(&Pv,~0ULL);
    bv_clear_unrolled(&Mv);
    bv_clear_unrolled(&Ph);
    bv_clear_unrolled(&Mh);
    bv_clear_unrolled(&Xp);
    int score=query_length;
    *lowest=score;
    *lowest_index_local=-1;

    for(int j=0;j<reference_length;++j){
        const bv_t *Eqc=&Eq[(unsigned char)reference[j]];
        bv_or_unrolled(&Xv,Eqc,&Mv);
        bv_and_unrolled(&tmp1,&Xv,&Pv);
        bv_add_unrolled(&tmp2,&tmp1,&Pv);
        bv_xor_unrolled(&tmp3,&tmp2,&Pv);
        bv_or_unrolled(&tmp4,&tmp3,&Xv);
        bv_or_unrolled(&Ph,&Mv,&tmp4);
        bv_and_unrolled(&Mh,&Pv,&Xv);
        if(bv_test_top_unrolled(&Ph,query_length))++score;
        if(bv_test_top_unrolled(&Mh,query_length))--score;
        if(score<*lowest){*lowest=score;*lowest_index_local=j;}
        bv_copy_unrolled(&Xp,&Xv);
    }
    return score;
}

// ---------------- Kernel ----------------
__global__ void levenshtein_kernel_shared_agg(
    int num_queries, int num_chunks, int num_orig_refs,
    const char *d_queries, const int *d_q_lens, const bv_t *d_Eq_queries,
    const char *d_refs, const int *d_ref_lens,
    const int *d_chunk_starts, const int *d_chunk_to_orig,
    const int *d_orig_ref_lens,
    int *d_pair_distances, int *d_lowest_score_orig, int *d_lowest_index_orig)
{
    extern __shared__ bv_t s_Eq[];
    int q=blockIdx.x;
    if(q>=num_queries)return;
    int tid=threadIdx.x;
    for(int i=tid;i<256;i+=blockDim.x) s_Eq[i]=d_Eq_queries[q*256+i];
    __syncthreads();

    int qlen=d_q_lens[q];
    for(int c=tid;c<num_chunks;c+=blockDim.x){
        const char *refptr=&d_refs[c*MAX_LENGTH];
        int rlen=d_ref_lens[c];
        int chunk_start=d_chunk_starts[c];
        int orig=d_chunk_to_orig[c];
        long long orig_idx=q*num_orig_refs+orig;
        int local_lowest_val,local_lowest_idx;
        int dist=bit_vector_levenshtein_local(qlen,refptr,rlen,s_Eq,&local_lowest_val,&local_lowest_idx);
        d_pair_distances[q*num_chunks+c]=dist;
        if(local_lowest_idx>=0){
            int global_idx=chunk_start+local_lowest_idx;
            int old;
            do{
                old=d_lowest_score_orig[orig_idx];
                if(local_lowest_val<old){
                    int res=atomicCAS(&d_lowest_score_orig[orig_idx],old,local_lowest_val);
                    if(res==old){
                        atomicExch(&d_lowest_index_orig[orig_idx],global_idx);
                        break;
                    }
                }else break;
            }while(true);
        }
    }
}

// ---------------- C API ----------------
extern "C" ALIGNMENT_API int align_sequences_gpu(
    const char *h_query,int q_len,const char *h_reference,int r_len,
    int *out_lowest_score,int *out_lowest_index,int *out_last_score)
{
    if(q_len<=0||r_len<=0||q_len>64*BV_WORDS)return -1;

    int num_queries=1,num_chunks=1,num_orig_refs=1;
    bv_t *h_Eq_queries=(bv_t*)calloc(256,sizeof(bv_t));
    if(!h_Eq_queries)return -2;
    for(int i=0;i<q_len;++i){
        unsigned char ch=(unsigned char)h_query[i];
        int word=i/64,bit=i%64;
        h_Eq_queries[ch].w[word]|=(1ULL<<bit);
    }

    char *h_queries=(char*)malloc(q_len);
    char *h_refs=(char*)malloc(r_len);
    memcpy(h_queries,h_query,q_len);
    memcpy(h_refs,h_reference,r_len);

    int h_q_lens[1]={q_len};
    int h_ref_lens[1]={r_len};
    int h_chunk_starts[1]={0};
    int h_chunk_to_orig[1]={0};
    int h_orig_ref_lens[1]={r_len};

    bv_t *d_Eq_queries; char *d_queries; char *d_refs;
    int *d_q_lens,*d_ref_lens,*d_chunk_starts,*d_chunk_to_orig,*d_orig_ref_lens;
    int *d_pair_distances,*d_lowest_score_orig,*d_lowest_index_orig;

    CUDA_CHECK(cudaMalloc(&d_Eq_queries,256*sizeof(bv_t)));
    CUDA_CHECK(cudaMalloc(&d_queries,q_len));
    CUDA_CHECK(cudaMalloc(&d_refs,r_len));
    CUDA_CHECK(cudaMalloc(&d_q_lens,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ref_lens,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chunk_starts,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chunk_to_orig,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_orig_ref_lens,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_distances,num_chunks*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lowest_score_orig,num_orig_refs*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lowest_index_orig,num_orig_refs*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_Eq_queries,h_Eq_queries,256*sizeof(bv_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries,h_queries,q_len,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_refs,h_refs,r_len,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_lens,h_q_lens,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_lens,h_ref_lens,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_starts,h_chunk_starts,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_to_orig,h_chunk_to_orig,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig_ref_lens,h_orig_ref_lens,sizeof(int),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_lowest_score_orig,0x7f,num_orig_refs*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lowest_index_orig,0xff,num_orig_refs*sizeof(int)));

    size_t shared_bytes=256*sizeof(bv_t);
    levenshtein_kernel_shared_agg<<<num_queries,threadsPerBlock,shared_bytes>>>(
        num_queries,num_chunks,num_orig_refs,
        d_queries,d_q_lens,d_Eq_queries,
        d_refs,d_ref_lens,
        d_chunk_starts,d_chunk_to_orig,
        d_orig_ref_lens,
        d_pair_distances,d_lowest_score_orig,d_lowest_index_orig);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_lowest_score,d_lowest_score_orig,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_lowest_index,d_lowest_index_orig,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_last_score,d_pair_distances,sizeof(int),cudaMemcpyDeviceToHost));

    cudaFree(d_Eq_queries); cudaFree(d_queries); cudaFree(d_refs);
    cudaFree(d_q_lens); cudaFree(d_ref_lens); cudaFree(d_chunk_starts);
    cudaFree(d_chunk_to_orig); cudaFree(d_orig_ref_lens);
    cudaFree(d_pair_distances); cudaFree(d_lowest_score_orig); cudaFree(d_lowest_index_orig);
    free(h_Eq_queries); free(h_queries); free(h_refs);
    return 0;
}
