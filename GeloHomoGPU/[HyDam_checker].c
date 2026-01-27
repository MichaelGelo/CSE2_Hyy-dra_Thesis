#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "hyrro_io.h"
#include <sys/time.h>

#define QUERY_FILE "que1_256.fasta"
#define REFERENCE_FILE "Stiff Brome.fasta"
#define LOOP_COUNT 1

typedef uint64_t u64;
#define BV_WORDS 4   // 256 bits
typedef struct { u64 w[BV_WORDS]; } BV;

/* --- BV helpers --- */
static inline void bv_zero(BV* a){for(int i=0;i<BV_WORDS;i++) a->w[i]=0;}
static inline void bv_set_all(BV* a){for(int i=0;i<BV_WORDS;i++) a->w[i]=~0ULL;}
static inline void bv_copy(const BV* s,BV* d){for(int i=0;i<BV_WORDS;i++) d->w[i]=s->w[i];}
static inline void bv_or(const BV* a,const BV* b,BV* r){for(int i=0;i<BV_WORDS;i++) r->w[i]=a->w[i]|b->w[i];}
static inline void bv_and(const BV* a,const BV* b,BV* r){for(int i=0;i<BV_WORDS;i++) r->w[i]=a->w[i]&b->w[i];}
static inline void bv_xor(const BV* a,const BV* b,BV* r){for(int i=0;i<BV_WORDS;i++) r->w[i]=a->w[i]^b->w[i];}
static inline void bv_not(const BV* a,BV* r){for(int i=0;i<BV_WORDS;i++) r->w[i]=~a->w[i];}
static inline void bv_shl1(const BV* a,BV* r){u64 c=0; for(int i=0;i<BV_WORDS;i++){u64 nc=a->w[i]>>63; r->w[i]=(a->w[i]<<1)|c; c=nc;}}
static inline void bv_add(const BV* a,const BV* b,BV* r){unsigned __int128 c=0; for(int i=0;i<BV_WORDS;i++){unsigned __int128 s=(unsigned __int128)a->w[i]+b->w[i]+c; r->w[i]=(u64)s; c=s>>64;}}
static inline int bv_test_msb(const BV* v,int m){int w=(m-1)/64,b=(m-1)%64; return (v->w[w]>>b)&1;}
static inline void bv_mask_top(BV* v,int m){if(m>=BV_WORDS*64) return; int w=(m-1)/64,b=(m-1)%64; u64 mask=(b==63)?~0ULL:((1ULL<<(b+1))-1); for(int i=w+1;i<BV_WORDS;i++) v->w[i]=0; v->w[w]&=mask;}

/* --- Hyyrö-Damerau bit-parallel algorithm --- */
int bit_vector_damerau(const char* query,const char* ref,int* scores,
                       int** hit_idxs,int* hit_count,
                       int** low_idxs,int* low_count,int* lowest_score){
    int m=(int)strlen(query), n=(int)strlen(ref);
    if(m>BV_WORDS*64){printf("Query too long\n"); return -1;}

    // Preprocess: Build Eq bit vectors (PEq in pseudocode)
    BV Eq[256]; 
    for(int i=0;i<256;i++) bv_zero(&Eq[i]);
    for(int i=0;i<m;i++){
        unsigned char c=(unsigned char)query[i]; 
        Eq[c].w[i/64]|=1ULL<<(i%64);
    }

    // Initialize bit vectors (line 2-5 in pseudocode)
    BV Pv, Mv, Ph, Mh, Xv, Xh, Xp;
    BV tmp, tmp2, addtmp, Xh_or_Pv, not_Xh_or_Pv;
    
    // Pre-compute mask once (OPTIMIZATION: avoid recomputing in loop)
    BV queryMask;
    bv_set_all(&queryMask);
    bv_mask_top(&queryMask, m);
    
    bv_set_all(&Pv);  // Pv = 1^m
    bv_and(&Pv, &queryMask, &Pv);  // Apply mask once
    bv_zero(&Mv);     // Mv = 0^m
    bv_zero(&Xp);     // Xp starts at 0
    
    int score=m;  // Score = m
    *hit_count=0; *low_count=0; *lowest_score=m;
    *hit_idxs=malloc(n*sizeof(int));
    *low_idxs=malloc(n*sizeof(int));

    // Main loop (line 6-18 in pseudocode)
    for(int j=0;j<n;j++){
        unsigned char rc=(unsigned char)ref[j];
        
        // Line 7: Eq = PEq[Σ[T[j]]]
        // Line 8: Xv = Eq | Mv
        bv_or(&Eq[rc], &Mv, &Xv);
        
        // Line 9: Xh = (((~Xh) & Xv) << 1) & Xp
        // Note: On first iteration, Xh is undefined, but the & Xp (which is 0) makes it 0
        bv_not(&Xh, &tmp);
        bv_and(&tmp, &Xv, &tmp2);
        bv_shl1(&tmp2, &tmp);
        bv_and(&tmp, &Xp, &Xh);
        
        // Line 10: Xh = Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv
        bv_and(&Xv, &Pv, &tmp);
        bv_add(&tmp, &Pv, &addtmp);
        bv_xor(&addtmp, &Pv, &tmp2);
        bv_or(&Xh, &tmp2, &Xh);
        bv_or(&Xh, &Xv, &Xh);
        bv_or(&Xh, &Mv, &Xh);
        
        // Line 11: Ph = Mv | ~(Xh | Pv)
        bv_or(&Xh, &Pv, &Xh_or_Pv);
        bv_not(&Xh_or_Pv, &not_Xh_or_Pv);
        bv_or(&Mv, &not_Xh_or_Pv, &Ph);
        
        // Line 12: Mh = Xh & Pv
        bv_and(&Xh, &Pv, &Mh);
        
        // Line 13: Xp = Xv (store old pattern bit-vector)
        bv_copy(&Xv, &Xp);
        
        // Line 14-15: Update score
        if(bv_test_msb(&Ph, m)) {
            score++;
        } else if(bv_test_msb(&Mh, m)) {
            score--;
        }
        
        scores[j]=score;

        // Track hits and lowest scores
        if(score==0) (*hit_idxs)[(*hit_count)++] = j;
        if(score<*lowest_score){
            *lowest_score=score;
            *low_count=1;
            (*low_idxs)[0]=j;
        } else if(score==*lowest_score){
            (*low_idxs)[(*low_count)++] = j;
        }

        // Line 16: Xv = (Ph << 1)
        bv_shl1(&Ph, &Xv);
        
        // Line 17: Pv = (Mh << 1) | ~(Xh | Xv)
        BV Mh_shl, Xh_or_Xv, not_Xh_or_Xv;
        bv_shl1(&Mh, &Mh_shl);
        bv_or(&Xh, &Xv, &Xh_or_Xv);
        bv_not(&Xh_or_Xv, &not_Xh_or_Xv);
        bv_or(&Mh_shl, &not_Xh_or_Xv, &Pv);
        
        // Line 18: Mv = Xh & Xv
        bv_and(&Xh, &Xv, &Mv);
        
        // Mask to pattern length (use pre-computed mask)
        bv_and(&Pv, &queryMask, &Pv);
        bv_and(&Mv, &queryMask, &Mv);
    }
    
    return score;
}

int main(){
    int num_queries=0,num_refs=0;
    char **queries=parse_fasta_file(QUERY_FILE,&num_queries);
    char **refs=parse_fasta_file(REFERENCE_FILE,&num_refs);
    if(!queries || !refs){fprintf(stderr,"Failed to load sequences\n"); return 1;}

    double total_time=0.0;

    for(int loop=0;loop<LOOP_COUNT;loop++){
        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);
        
        for(int q=0;q<num_queries;q++){
            for(int r=0;r<num_refs;r++){
                int n=(int)strlen(refs[r]);
                int* scores=malloc(n*sizeof(int));
                int *hit_idxs=NULL, hit_count=0;
                int *low_idxs=NULL, low_count=0, lowest=INT_MAX;

                int dist=bit_vector_damerau(queries[q],refs[r],scores,
                                           &hit_idxs,&hit_count,
                                           &low_idxs,&low_count,&lowest);

                if(loop==0){
                    printf("----------------------------------------------------------------------------\n");
                    printf("Pair: Q%d(%d) Vs R%d(%d)\n",q+1,(int)strlen(queries[q]),r+1,(int)strlen(refs[r]));
                    printf("Number of Hits: %d\n",hit_count);
                    if(hit_count>0){
                        printf("Hit Indexes: [");
                        for(int i=0;i<hit_count;i++){printf("%d",hit_idxs[i]); if(i<hit_count-1) printf(", ");} printf("]\n");
                        printf("Lowest Score: N/A\n");
                        printf("Lowest Score Indexes: N/A\n");
                    } else {
                        printf("Hit Indexes: N/A\n");
                        if(lowest!=INT_MAX){
                            printf("Lowest Score: %d\n",lowest);
                            printf("Lowest Score Indexes: [");
                            for(int i=0;i<low_count;i++){printf("%d",low_idxs[i]); if(i<low_count-1) printf(", ");} printf("]\n");
                        } else {
                            printf("Lowest Score: N/A\nLowest Score Indexes: N/A\n");
                        }
                    }
                    if(n>0) printf("Last Score: %d\n", scores[n-1]);
                    else printf("Last Score: N/A\n");
                    printf("----------------------------------------------------------------------------\n\n");
                }
                free(scores); free(hit_idxs); free(low_idxs);
            }
        }
        
        gettimeofday(&t_end, NULL);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_usec - t_start.tv_usec) / 1e6;
        total_time += elapsed;
    }

    printf("%d loop Average time: %.6f sec.\n", LOOP_COUNT, total_time/LOOP_COUNT);

    for(int i=0;i<num_queries;i++) free(queries[i]); free(queries);
    for(int i=0;i<num_refs;i++) free(refs[i]); free(refs);
    return 0;
}
