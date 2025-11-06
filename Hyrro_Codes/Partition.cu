
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "C_utils.h"

#define reference_file "que5_256.fasta"
#define query_file "ref5_50M.fasta"

void write_fasta(FILE *file, const char *sequence, int line_width){
    int length = strlen(sequence);
    for (int i = 0; i < length; i += line_width) {
        fprintf(file, "%.*s\n", line_width, sequence + i);
    }
}

int main() {
    int num_references = 0;
    char **reference_seqs = parse_fasta_file(reference_file, &num_references);

    const char *query = read_file_into_string(query_file);
    int q_length = strlen(query);
    int p = 10700;
    int reference_length = 0;

    FILE *output_file = fopen("partition.fasta", "w");
    if (output_file == NULL) {
        perror("Unable to open output file");
        return 1;
    }

    int global_batch = 0;  // global batch counter

    for (int i = 0; i < num_references; i++) {
        char *sequence = reference_seqs[i];
        reference_length = strlen(sequence);

        int previous_k = 0;

        while (previous_k < reference_length) {
            int chunk_size = p;
            if (previous_k + p > reference_length) {
                chunk_size = reference_length - previous_k;
            }

            int extended_chunk_size = chunk_size + (q_length - 1);
            if (previous_k + extended_chunk_size > reference_length) {
                extended_chunk_size = reference_length - previous_k;
            }

            char ref_toparti[extended_chunk_size + 1];
            for (int k = 0; k < extended_chunk_size; k++) {
                ref_toparti[k] = sequence[previous_k + k];
            }
            ref_toparti[extended_chunk_size] = '\0';

            // print to terminal with chunk size and number
            printf("Chunk No: %d, Chunk Size: %d\n", global_batch, extended_chunk_size);
            printf("Batch %d, Reference %d: %s\n", global_batch, i, ref_toparti);

            // write to output FASTA
            fprintf(output_file, ">Batch %d, Reference %d\n", global_batch, i);
            write_fasta(output_file, ref_toparti, 60);

            previous_k += p;
            global_batch++;
        }
    }

    fclose(output_file);
    printf("Partitioned FASTA file written.\n");

    // free memory
    for (int i = 0; i < num_references; i++) {
        free(reference_seqs[i]);
    }
    free(reference_seqs);
    free((void*)query);

    return 0;
}
