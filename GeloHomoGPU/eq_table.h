// eq_table.h
// Construction of Eq (equality) tables for bit-vector Levenshtein algorithm

#ifndef EQ_TABLE_H
#define EQ_TABLE_H

#include "bitvector.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// BUILD EQ TABLES FOR ALL QUERIES
// Creates bit-vector lookup tables where Eq[c] marks positions where
// character c appears in the query string. Used for fast character matching
// during Levenshtein computation.
// Returns array of size (numQueries * 256) containing one 256-entry table per query
// ============================================================================
static inline bv_t* buildEqTables(char** queries, int* queryLengths, int numQueries) {
    bv_t* eqTables = (bv_t*)malloc((size_t)numQueries * 256 * sizeof(bv_t));
    if (!eqTables) {
        fprintf(stderr, "ERROR: Out of memory for Eq tables\n");
        return NULL;
    }
    
    memset(eqTables, 0, (size_t)numQueries * 256 * sizeof(bv_t));

    for (int q = 0; q < numQueries; ++q) {
        int qlen = queryLengths[q];
        const char* queryStr = queries[q];
        
        for (int i = 0; i < qlen; ++i) {
            unsigned char ch = (unsigned char)queryStr[i];
            int word = i / 64;
            int bit = i % 64;
            eqTables[(long long)q * 256 + ch].w[word] |= (1ULL << bit);
        }
    }

    return eqTables;
}

#endif // EQ_TABLE_H