// hyrro_io.h
// Implementation of FASTA file parsing (moved from C_utils.c)

#ifndef HYRRO_IO_H
#define HYRRO_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_IO_LENGTH (1 << 30)  // 536.8MB max file size
#define MAX_LINE_LENGTH (1 << 14)
#define MAX_FILENAME 512


#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// READ ENTIRE FILE INTO STRING
// ============================================================================
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
// PARSE FASTA FILE INTO ARRAY OF SEQUENCES (OPTIMIZED)
// ============================================================================
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
    size_t current_seq_capacity = 0;
    char line[MAX_LINE_LENGTH];
    
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '>') {
            // New sequence header
            if (current_seq != NULL && current_seq_len > 0) {
                sequences = (char**)realloc(sequences, (seq_count + 1) * sizeof(char*));
                sequences[seq_count] = (char*)malloc(current_seq_len + 1);
                memcpy(sequences[seq_count], current_seq, current_seq_len);
                sequences[seq_count][current_seq_len] = '\0';
                seq_count++;
            }
            // Reset for new sequence - pre-allocate 256MB
            if (current_seq) free(current_seq);
            current_seq_capacity = MAX_IO_LENGTH;
            current_seq = (char*)malloc(current_seq_capacity);
            current_seq_len = 0;
        } else {
            // Sequence line
            size_t line_len = strlen(line);
            while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
                line_len--;
            }
            
            if (line_len > 0 && current_seq != NULL) {
                if (current_seq_len + line_len > MAX_IO_LENGTH) {
                    line_len = MAX_IO_LENGTH - current_seq_len;
                }
                memcpy(current_seq + current_seq_len, line, line_len);
                current_seq_len += line_len;
            }
        }
    }
    
    // Handle last sequence
    if (current_seq != NULL && current_seq_len > 0) {
        sequences = (char**)realloc(sequences, (seq_count + 1) * sizeof(char*));
        sequences[seq_count] = (char*)malloc(current_seq_len + 1);
        memcpy(sequences[seq_count], current_seq, current_seq_len);
        sequences[seq_count][current_seq_len] = '\0';
        seq_count++;
    }
    
    if (current_seq) free(current_seq);
    fclose(file);
    *num_sequences = seq_count;
    return sequences;
}

// ============================================================================
// FOLDER SCANNING - GET ALL FASTA FILES FROM A DIRECTORY
// ============================================================================
typedef struct {
    char** filenames;  // dynamically allocated array of path strings
    int count;
} FastaFileList;

static inline int ends_with_fasta(const char* filename) {
    size_t len = strlen(filename);
    if (len < 6) return 0;
    return (strcmp(&filename[len-6], ".fasta") == 0) ||
           (strcmp(&filename[len-4], ".fas") == 0) ||
           (strcmp(&filename[len-3], ".fa") == 0);
}

static inline FastaFileList scan_folder_for_fasta(const char* folder_path) {
    FastaFileList result = {0};

    DIR *dir = opendir(folder_path);
    if (!dir) {
        fprintf(stderr, "ERROR: Failed to open directory: %s\n", folder_path);
        return result;
    }

    // First pass: count matching files
    struct dirent *entry;
    int n = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && ends_with_fasta(entry->d_name)) n++;
    }

    if (n == 0) {
        closedir(dir);
        return result;
    }

    result.filenames = (char**)malloc(n * sizeof(char*));
    if (!result.filenames) {
        fprintf(stderr, "ERROR: Out of memory scanning %s\n", folder_path);
        closedir(dir);
        return result;
    }

    // Second pass: collect full paths
    rewinddir(dir);
    int i = 0;
    while ((entry = readdir(dir)) != NULL && i < n) {
        if (entry->d_type == DT_REG && ends_with_fasta(entry->d_name)) {
            char buf[MAX_FILENAME];
            snprintf(buf, MAX_FILENAME, "%s/%s", folder_path, entry->d_name);
            result.filenames[i++] = strdup(buf);
        }
    }
    closedir(dir);
    result.count = i;

    // Sort alphabetically for deterministic ordering
    for (int a = 0; a < i - 1; a++) {
        for (int b = a + 1; b < i; b++) {
            if (strcmp(result.filenames[a], result.filenames[b]) > 0) {
                char* tmp = result.filenames[a];
                result.filenames[a] = result.filenames[b];
                result.filenames[b] = tmp;
            }
        }
    }

    return result;
}

static inline void free_fasta_file_list(FastaFileList* list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) free(list->filenames[i]);
    free(list->filenames);
    list->filenames = NULL;
    list->count = 0;
}

// ============================================================================
// RESULT FOLDER CREATION
// ============================================================================
static inline int create_results_folder(const char* folder_path) {
    struct stat st = {0};
    
    if (stat(folder_path, &st) == -1) {
        // Folder doesn't exist, create it
        if (mkdir(folder_path, 0755) == -1) {
            fprintf(stderr, "ERROR: Failed to create results folder: %s\n", folder_path);
            return 0;
        }
        printf("Created results folder: %s\n", folder_path);
    }
    
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif // HYYRO_IO_H