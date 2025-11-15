#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "C_utils.h"

#define MAX_LENGTH (1 << 28) // Increased from 2^24 (16.7M) to 2^28 (268.4M)
#define MAX_LINE_LENGTH (1 << 14)

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
