/**
 * @file cpu_utils.c
 * @brief Implementation of CPU utility functions
 * 
 * This file implements the FASTA parsing functions declared in cpu_utils.h
 */

#include "cpu_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH (1 << 14)  // 16KB per line
#define MAX_SEQ_LENGTH (1 << 29)   // 536MB max sequence length (supports up to 500M files)

// ============================================================================
// FILE READING
// ============================================================================

/**
 * @brief Read entire file into string
 */
char* read_file_into_string(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Allocate buffer
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        perror("Failed to allocate memory");
        fclose(file);
        return NULL;
    }

    // Read file
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != (size_t)file_size) {
        perror("Failed to read the file completely");
        free(buffer);
        fclose(file);
        return NULL;
    }

    buffer[file_size] = '\0';
    fclose(file);
    return buffer;
}

// ============================================================================
// FASTA PARSING
// ============================================================================

/**
 * @brief Parse FASTA file and extract sequences
 */
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
            // === Header line: save previous sequence and start new one ===
            if (current_seq != NULL) {
                if (current_seq_len > 0) {
                    // Allocate space for sequence in array
                    sequences = (char**)realloc(sequences, (seq_count + 1) * sizeof(char*));
                    sequences[seq_count] = (char*)malloc(current_seq_len + 1);
                    
                    // Copy sequence and null-terminate
                    memcpy(sequences[seq_count], current_seq, current_seq_len);
                    sequences[seq_count][current_seq_len] = '\0';
                    seq_count++;
                }
                
                // Free temporary buffer
                free(current_seq);
                current_seq = NULL;
                current_seq_len = 0;
            }
        } else {
            // === Sequence data line ===
            size_t line_len = strlen(line);
            
            // Remove trailing newline/carriage return
            while (line_len > 0 && 
                   (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
                line_len--;
            }

            // Check if adding this line would exceed max length
            if (current_seq_len + line_len > MAX_SEQ_LENGTH) {
                line_len = MAX_SEQ_LENGTH - current_seq_len;
            }

            // Append line to current sequence
            if (line_len > 0) {
                current_seq = (char*)realloc(current_seq, current_seq_len + line_len + 1);
                memcpy(current_seq + current_seq_len, line, line_len);
                current_seq_len += line_len;
                current_seq[current_seq_len] = '\0';
            }
        }
    }

    // === Save last sequence ===
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