// data_loader.h
// Functions for loading FASTA files and preparing data for GPU processing

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string.h>
#include <stdio.h>
#include "config.h"

// External C functions from C_utils.h
#ifdef __cplusplus
extern "C" {
#endif

extern char* read_file_into_string(const char* filename);
extern char** parse_fasta_file(const char *filename, int *num_sequences);

#ifdef __cplusplus
}
#endif

// ============================================================================
// LOAD QUERIES FROM FASTA FILE
// Reads query sequences from QUERY_FILE using FASTA parser
// Returns array of query strings and sets numQueries
// ============================================================================
static inline char** loadQueries(int *numQueries) {
    return parse_fasta_file(QUERY_FILE, numQueries);
}

// ============================================================================
// LOAD REFERENCES FROM FASTA FILE
// Reads reference sequences from REFERENCE_FILE
// Returns array of reference strings and their lengths
// ============================================================================
static inline char** loadReferences(int *numRefs, int **refLens) {
    int num = 0;
    char **refs = parse_fasta_file(REFERENCE_FILE, &num);
    
    if (!refs || num <= 0) {
        *numRefs = 0;
        *refLens = NULL;
        return NULL;
    }

    int *lens = (int*)malloc(sizeof(int) * num);
    if (!lens) {
        fprintf(stderr, "ERROR: Out of memory for reference lengths\n");
        for (int i = 0; i < num; ++i) free(refs[i]);
        free(refs);
        *numRefs = 0;
        *refLens = NULL;
        return NULL;
    }

    for (int i = 0; i < num; ++i) {
        lens[i] = (int)strlen(refs[i]);
    }

    *numRefs = num;
    *refLens = lens;
    return refs;
}

// ============================================================================
// COMPUTE QUERY LENGTHS
// Extracts lengths of all query sequences into an array
// ============================================================================
static inline int* computeQueryLengths(char** queries, int numQueries) {
    int* lengths = (int*)malloc(numQueries * sizeof(int));
    if (!lengths) {
        fprintf(stderr, "ERROR: Out of memory for query lengths\n");
        return NULL;
    }
    
    for (int q = 0; q < numQueries; ++q) {
        lengths[q] = (int)strlen(queries[q]);
    }
    
    return lengths;
}

#endif // DATA_LOADER_H