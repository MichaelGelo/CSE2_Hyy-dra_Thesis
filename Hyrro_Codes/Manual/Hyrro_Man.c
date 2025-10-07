#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#define MAX_LENGTH (1 << 24)
#define MAX_LINE_LENGTH (1 << 14)

#define query_file "D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/Testing/longquery64.fasta"
#define reference_file "D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/Testing/longref128.fasta"

#define loope 10
typedef uint64_t bitvector;

FILE *out = NULL;  // global file pointer

// ✅ macro defined here, before any functions
#define LOG(fmt, ...) \
    do { \
        printf(fmt, ##__VA_ARGS__); \
        if (out) fprintf(out, fmt, ##__VA_ARGS__); \
    } while (0)
char* read_file_into_string(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        perror("Failed to allocate memory");
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        perror("Failed to read the file completely");
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[file_size] = '\0';
    fclose(file);
    return buffer;
}

char** parse_fasta_file(const char *filename, int *num_sequences) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open FASTA file");
        return NULL;
    }
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
            while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
                line_len--;
            }
            if (current_seq_len + line_len > MAX_LENGTH) {
                line_len = MAX_LENGTH - current_seq_len;
            }
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

void print_bits(unsigned long long x, int m) {
    char buffer[1024];  // enough for most cases (adjust if m > 1024)
    int pos = 0;

    for (int i = m - 1; i >= 0; i--) {
        buffer[pos++] = (x & (1ULL << i)) ? '1' : '0';

        // Add space after every 4 bits (but not after the last group)
        if (i % 4 == 0 && i != 0) {
            buffer[pos++] = ' ';
        }
    }
    buffer[pos] = '\0';

    LOG("%s", buffer);
}



int bit_vector_levenshtein(const char *query, const char *reference, int *scores) {
    int m = strlen(query);
    int n = strlen(reference);
    if (m > MAX_LENGTH || n > MAX_LENGTH) {
       LOG("Error: Strings too long for this implementation!\n");
        return -1;
    }

    bitvector Pv = ~0ULL;   // line 4
    bitvector Mv = 0;       // line 5
    bitvector Eq[256] = {0};
    bitvector Ph = 0, Mh = 0, Xv = 0, Xh = 0, Xp = 0;

    // preprocess Eq
    for (int i = 0; i < m; i++) {
        Eq[(unsigned char)query[i]] |= (1ULL << i);
    }

    int score = m;          // line 3

    for (int j = 0; j < n; j++) {
       LOG("=== Iteration j=%d (reference[%d] = '%c') ===\n", j+1, j+1, reference[j]);

        bitvector Eqj = Eq[(unsigned char)reference[j]];
       LOG("Line 7  Eq: "); print_bits(Eqj, m);LOG("\n");
       LOG("Line 7  Pv: "); print_bits(Pv, m);LOG("\n");
       LOG("Line 7  Mv: "); print_bits(Mv, m);LOG("\n");
       LOG("Line 7  Xv: "); print_bits(Xv, m);LOG("\n");
       LOG("Line 7  Xh: "); print_bits(Xh, m);LOG("\n");
       LOG("Line 7  Ph: "); print_bits(Ph, m);LOG("\n");
       LOG("Line 7  Mh: "); print_bits(Mh, m);LOG("\n");
       LOG("Line 7  Xp: "); print_bits(Xp, m);LOG("\n");
       LOG("Previous Score: %d\n", score);


        Xv = Eqj | Mv;
       LOG("Line 8  Xv: "); print_bits(Xv, m);LOG("\n");

        Xh = (((~Xh) & Xv) << 1) & Xp;
        Xh = Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv;
       LOG("Line 9-10 Xh: "); print_bits(Xh, m);LOG("\n");

        Ph = Mv | ~(Xh | Pv);
       LOG("Line 11 Ph: "); print_bits(Ph, m);LOG("\n");

        Mh = Xh & Pv;
       LOG("Line 12 Mh: "); print_bits(Mh, m);LOG("\n");

        Xp = Xv;
       LOG("Line 13 Xp: "); print_bits(Xp, m);LOG("\n");

        if (Ph & (1ULL << (m - 1))) {
            score++;
           LOG("Line 14 Score incremented -> %d\n", score);
        } else if (Mh & (1ULL << (m - 1))) {
            score--;
           LOG("Line 15 Score decremented -> %d\n", score);
        }

        Xv = (Ph << 1);
       LOG("Line 16 Xv: "); print_bits(Xv, m);LOG("\n");

        Pv = (Mh << 1) | ~(Xh | Xv);
       LOG("Line 17 Pv: "); print_bits(Pv, m);LOG("\n");

        Mv = Xh & Xv;
       LOG("Line 18 Mv: "); print_bits(Mv, m);LOG("\n");

       LOG("Current Score: %d\n", score);
       LOG("-----------------------------\n");

        if (scores) scores[j] = score;
    }

    return score;
}


int main() {
    out = fopen("D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/output.txt", "w");
    if (!out) {
        perror("Failed to open output.txt");
        return 1;
    }
    int num_queries = 0, num_references = 0;
    char **query_seqs = parse_fasta_file(query_file, &num_queries);
    char **reference_seqs = parse_fasta_file(reference_file, &num_references);

    if (!query_seqs || !reference_seqs) {
        fprintf(stderr, "Error: Failed to load sequences.\n");
        return 1;
    }

    for (int q = 0; q < num_queries; q++) {
        for (int r = 0; r < num_references; r++) {
            int n = strlen(reference_seqs[r]);
            int *scores = (int*)malloc(n * sizeof(int));
            if (!scores) {
                perror("Failed to allocate scores");
                continue;
            }

            LARGE_INTEGER freq, t1, t2;
            QueryPerformanceFrequency(&freq);

            QueryPerformanceCounter(&t1);
            int final_distance = bit_vector_levenshtein(query_seqs[q], reference_seqs[r], scores);
            QueryPerformanceCounter(&t2);

            double elapsed = (double)(t2.QuadPart - t1.QuadPart) / freq.QuadPart;

            // --- Print Output (matching Python style) ---
           LOG("Query: %s\n", query_seqs[q]);
           LOG("Text:  %s\n", reference_seqs[r]);
           LOG("Length of query(p) is %zu\n", strlen(query_seqs[q]));
           LOG("Length of text(t) is %zu\n", strlen(reference_seqs[r]));
           LOG("score:\n");
           LOG("%s\n", reference_seqs[r]);
            for (int i = 0; i < n; i++) {
               LOG("%d", scores[i]);
                if (i < n - 1)LOG(",");
            }
           LOG("\n\nExecution time:  %.15f sec.\n", elapsed);
           LOG("Done...\n\n");
            
            free(scores);
        }
    }
    
    for (int i = 0; i < num_queries; i++) free(query_seqs[i]);
    free(query_seqs);
    for (int i = 0; i < num_references; i++) free(reference_seqs[i]);
    free(reference_seqs);
    
    if (out) fclose(out);
    return 0;
}