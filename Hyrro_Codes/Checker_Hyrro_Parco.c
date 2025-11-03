#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <limits.h>

#define MAX_LENGTH (1 << 24)
#define MAX_LINE_LENGTH (1 << 14)

#define query_file "D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/Testing/temp-que.fasta"
#define reference_file "D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/Testing/temp-ref.fasta"

#define loope 10
typedef uint64_t u64;

#define BV_WORDS 4   /* 4 * 64 = 256 bits */
typedef struct { u64 w[BV_WORDS]; } BV;

/* --- FASTA reader (unchanged) --- */
char* read_file_into_string(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror("Failed to open file"); return NULL; }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) { perror("Failed to allocate memory"); fclose(file); return NULL; }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) { perror("Failed to read the file completely"); free(buffer); fclose(file); return NULL; }
    buffer[file_size] = '\0';
    fclose(file);
    return buffer;
}

char** parse_fasta_file(const char *filename, int *num_sequences) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("Failed to open FASTA file"); return NULL; }
    char **sequences = NULL;
    int seq_count = 0;
    char *current_seq = NULL;
    size_t current_seq_len = 0;
    char line[MAX_LINE_LENGTH];

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '>') {
            if (current_seq != NULL) {
                if (current_seq_len > 0) {
                    sequences = (char**)realloc(sequences, (seq_count + 1) * sizeof(char*));
                    sequences[seq_count] = (char*)malloc(current_seq_len + 1);
                    memcpy(sequences[seq_count], current_seq, current_seq_len);
                    sequences[seq_count][current_seq_len] = '\0';
                    seq_count++;
                }
                free(current_seq);
                current_seq = NULL;
                current_seq_len = 0;
            }
        } else {
            size_t line_len = strlen(line);
            while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) line_len--;
            if (current_seq_len + line_len > MAX_LENGTH) line_len = MAX_LENGTH - current_seq_len;
            if (line_len > 0) {
                current_seq = (char*)realloc(current_seq, current_seq_len + line_len + 1);
                memcpy(current_seq + current_seq_len, line, line_len);
                current_seq_len += line_len;
                current_seq[current_seq_len] = '\0';
            }
        }
    }
    if (current_seq != NULL && current_seq_len > 0) {
        sequences = (char**)realloc(sequences, (seq_count + 1) * sizeof(char*));
        sequences[seq_count] = (char*)malloc(current_seq_len + 1);
        memcpy(sequences[seq_count], current_seq, current_seq_len);
        sequences[seq_count][current_seq_len] = '\0';
        seq_count++;
        free(current_seq);
    }
    fclose(file);
    *num_sequences = seq_count;
    return sequences;
}

/* --- 256-bit BV helpers --- */
static inline void bv_zero(BV *a) { for (int i = 0; i < BV_WORDS; ++i) a->w[i] = 0ULL; }
static inline void bv_set_all(BV *a) { for (int i = 0; i < BV_WORDS; ++i) a->w[i] = ~0ULL; }
static inline void bv_copy(const BV *s, BV *d) { for (int i = 0; i < BV_WORDS; ++i) d->w[i] = s->w[i]; }
static inline void bv_or(const BV *a, const BV *b, BV *r) { for (int i = 0; i < BV_WORDS; ++i) r->w[i] = a->w[i] | b->w[i]; }
static inline void bv_and(const BV *a, const BV *b, BV *r) { for (int i = 0; i < BV_WORDS; ++i) r->w[i] = a->w[i] & b->w[i]; }
static inline void bv_xor(const BV *a, const BV *b, BV *r) { for (int i = 0; i < BV_WORDS; ++i) r->w[i] = a->w[i] ^ b->w[i]; }
static inline void bv_not(const BV *a, BV *r) { for (int i = 0; i < BV_WORDS; ++i) r->w[i] = ~a->w[i]; }

/* left shift by 1 (toward higher indices) */
static inline void bv_shl1(const BV *a, BV *r) {
    u64 carry = 0ULL;
    for (int i = 0; i < BV_WORDS; ++i) {
        u64 new_carry = (a->w[i] >> 63) & 1ULL;
        r->w[i] = (a->w[i] << 1) | carry;
        carry = new_carry;
    }
}

/* right shift by 1 (toward lower indices) */
static inline void bv_shr1(const BV *a, BV *r) {
    u64 carry = 0ULL;
    for (int i = BV_WORDS - 1; i >= 0; --i) {
        u64 new_carry = (a->w[i] & 1ULL);
        r->w[i] = (a->w[i] >> 1) | (carry << 63);
        carry = new_carry;
    }
}

/* r = a + b, return carry-out (0/1) */
static inline int bv_add(const BV *a, const BV *b, BV *r) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < BV_WORDS; ++i) {
        unsigned __int128 sum = (unsigned __int128)a->w[i] + b->w[i] + carry;
        r->w[i] = (u64)sum;
        carry = (sum >> 64) & 1;
    }
    return (int)carry;
}

/* out = (Xv & Pv) + Pv  (tmp is scratch) */
static inline void bv_and_add(const BV *Xv, const BV *Pv, BV *tmp, BV *out) {
    bv_and(Xv, Pv, tmp);
    bv_add(tmp, Pv, out);
}

/* mask bits >= m */
static inline void bv_mask_top(BV *v, int m) {
    if (m >= BV_WORDS * 64) return;
    int last_word = (m - 1) / 64;
    int last_bit = (m - 1) % 64;
    u64 last_mask = (last_bit == 63) ? ~0ULL : ((1ULL << (last_bit + 1)) - 1ULL);
    for (int i = last_word + 1; i < BV_WORDS; ++i) v->w[i] = 0ULL;
    v->w[last_word] &= last_mask;
}

/* test bit m-1 */
static inline int bv_test_msb(const BV *v, int m) {
    if (m == 0) return 0;
    int idx = (m - 1) / 64;
    int off = (m - 1) % 64;
    return ( (v->w[idx] >> off) & 1ULL ) ? 1 : 0;
}

/* --- Myers bit-parallel Levenshtein adapted to 256-bit BV --- */
/* scores[] must have size n (reference length) */
int bit_vector_levenshtein(const char *query, const char *reference, int *scores) {
    int m = (int)strlen(query);
    int n = (int)strlen(reference);
    if (m <= 0) return -1;
    if (m > BV_WORDS * 64) { printf("Error: query too long (max %d)\n", BV_WORDS * 64); return -1; }

    BV Pv, Mv, Xv, Xh, Ph, Mh, tmp, addtmp;
    BV Eq[256];
    for (int i = 0; i < 256; ++i) bv_zero(&Eq[i]);

    for (int i = 0; i < m; ++i) {
        unsigned char ch = (unsigned char)query[i];
        int word = i / 64;
        int bit = i % 64;
        Eq[ch].w[word] |= (1ULL << bit);
    }

    bv_set_all(&Pv);
    bv_zero(&Mv);
    bv_mask_top(&Pv, m);

    int score = m;
    for (int j = 0; j < n; ++j) {
        unsigned char rc = (unsigned char)reference[j];

        /* Xv = Eq[rc] | Mv */
        bv_or(&Eq[rc], &Mv, &Xv);

        /* Xh = (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv */
        bv_and_add(&Xv, &Pv, &tmp, &addtmp);  /* addtmp = (Xv & Pv) + Pv */
        bv_xor(&addtmp, &Pv, &Xh);            /* Xh = addtmp ^ Pv */
        BV t0; bv_or(&Xh, &Xv, &t0); bv_or(&t0, &Mv, &Xh);

        /* Ph = Mv | ~(Xh | Pv) */
        BV t1; bv_or(&Xh, &Pv, &t1); bv_not(&t1, &tmp); bv_or(&Mv, &tmp, &Ph);

        /* Mh = Xh & Pv */
        bv_and(&Xh, &Pv, &Mh);

        if (bv_test_msb(&Ph, m)) score++;
        if (bv_test_msb(&Mh, m)) score--;

        scores[j] = score;

        /* Pv = (Mh << 1) | ~(Xh | (Ph << 1)) */
        BV Mh_shl, Ph_shl, xh_or_phshl, not_xh_or_phshl;
        bv_shl1(&Mh, &Mh_shl);
        bv_shl1(&Ph, &Ph_shl);
        bv_or(&Xh, &Ph_shl, &xh_or_phshl);
        bv_not(&xh_or_phshl, &not_xh_or_phshl);
        BV newPv; bv_or(&Mh_shl, &not_xh_or_phshl, &newPv);

        /* Mv = Xh & (Ph << 1) */
        BV newMv; bv_and(&Xh, &Ph_shl, &newMv);

        bv_copy(&newPv, &Pv);
        bv_copy(&newMv, &Mv);

        bv_mask_top(&Pv, m);
        bv_mask_top(&Mv, m);
    }
    return score;
}

/* --- helper: format elapsed seconds into appropriate unit (s / ms / us / ns) --- */
static void format_time_auto(double seconds, char *buf, size_t buf_len) {
    if (seconds >= 1.0) {
        snprintf(buf, buf_len, "%.6f s", seconds);
    } else if (seconds >= 1e-3) {
        snprintf(buf, buf_len, "%.3f ms", seconds * 1e3);
    } else if (seconds >= 1e-6) {
        snprintf(buf, buf_len, "%.3f us", seconds * 1e6);
    } else {
        // show integer nanoseconds for very small durations
        snprintf(buf, buf_len, "%.0f ns", seconds * 1e9);
    }
}

/* --- main: iterate pairs and print requested output --- */
int main() {
    int num_queries = 0, num_references = 0;
    char **query_seqs = parse_fasta_file(query_file, &num_queries);
    char **reference_seqs = parse_fasta_file(reference_file, &num_references);

    if (!query_seqs || !reference_seqs) {
        fprintf(stderr, "Error: Failed to load sequences.\n");
        return 1;
    }

    FILE *out = fopen("output.txt", "w");
    if (!out) {
        perror("Failed to open output.txt for writing");
        out = stdout;
    }

    LARGE_INTEGER freq, t_start_all, t_end_all;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start_all);

    for (int q = 0; q < num_queries; q++) {
        for (int r = 0; r < num_references; r++) {
            int n = (int)strlen(reference_seqs[r]);
            int m = (int)strlen(query_seqs[q]);
            if (m > BV_WORDS * 64) {
                fprintf(stderr, "Query length %d exceeds BV capacity %d\n", m, BV_WORDS * 64);
                continue;
            }
            int *scores = (int*)malloc(n * sizeof(int));
            if (!scores) { perror("Failed to allocate scores"); continue; }

            LARGE_INTEGER t1, t2;
            QueryPerformanceCounter(&t1);
            int final_distance = bit_vector_levenshtein(query_seqs[q], reference_seqs[r], scores);
            QueryPerformanceCounter(&t2);

            double elapsed_pair = (double)(t2.QuadPart - t1.QuadPart) / freq.QuadPart;

            int *hit_idxs = (int*)malloc(n * sizeof(int));
            int hit_count = 0;
            int *low_idxs = (int*)malloc(n * sizeof(int));
            int low_count = 0;
            int lowest = INT_MAX;

            for (int i = 0; i < n; i++) {
                int s = scores[i];
                if (s == 0) hit_idxs[hit_count++] = i;
                if (s < lowest) { lowest = s; low_count = 0; low_idxs[low_count++] = i; }
                else if (s == lowest) low_idxs[low_count++] = i;
            }

            fprintf(out, "----------------------------------------------------------------------------\n");
            fprintf(out, "Pair: Q%d(%d) Vs R%d(%d)\n", q+1, m, r+1, n);
            fprintf(out, "Number of Hits: %d\n", hit_count);

            if (hit_count > 0) {
                fprintf(out, "Hit Indexes: [");
                for (int h = 0; h < hit_count; h++) { fprintf(out, "%d", hit_idxs[h]); if (h < hit_count - 1) fprintf(out, ", "); }
                fprintf(out, "]\n");
                fprintf(out, "Lowest Score: N/A\n");
                fprintf(out, "Lowest Score Indexes: N/A\n");
            } else {
                fprintf(out, "Hit Indexes: N/A\n");
                if (lowest != INT_MAX) {
                    fprintf(out, "Lowest Score: %d\n", lowest);
                    if (low_count > 0) {
                        fprintf(out, "Lowest Score Indexes: [");
                        for (int li = 0; li < low_count; li++) { fprintf(out, "%d", low_idxs[li]); if (li < low_count - 1) fprintf(out, ", "); }
                        fprintf(out, "]\n");
                    } else {
                        fprintf(out, "Lowest Score Indexes: N/A\n");
                    }
                } else {
                    fprintf(out, "Lowest Score: N/A\n");
                    fprintf(out, "Lowest Score Indexes: N/A\n");
                }
            }

            if (n > 0) fprintf(out, "Last Score: %d\n", scores[n-1]);
            else fprintf(out, "Last Score: N/A\n");

            fprintf(out, "----------------------------------------------------------------------------\n\n");

            free(scores);
            free(hit_idxs);
            free(low_idxs);
        }
    }

    QueryPerformanceCounter(&t_end_all);
    double elapsed_all = (double)(t_end_all.QuadPart - t_start_all.QuadPart) / freq.QuadPart;

    // print total execution time at the end, raw seconds (still available in human units if needed)
    fprintf(out, "Execution time: %.15f sec.\n", elapsed_all);

    for (int i = 0; i < num_queries; i++) free(query_seqs[i]);
    free(query_seqs);
    for (int i = 0; i < num_references; i++) free(reference_seqs[i]);
    free(reference_seqs);

    if (out != stdout) {
        fclose(out);
        printf("Done... output written to output.txt\n");
    } else {
        printf("Done...\n");
    }

    return 0;
}