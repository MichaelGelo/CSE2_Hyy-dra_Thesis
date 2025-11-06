
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "C_utils.h"

#define query_file "D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/Queries/que5_256.fasta"
#define reference_file "D:/download/College Life/Thesis/CSE2_Hyy-dra_Thesis/Resources/References/ref5_50M.fasta"

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

    FILE *output_file = fopen("partitioned_output.fasta", "w");
    if (output_file == NULL) {
        perror("Unable to open output file");
        return 1;
    }

    for (int i = 0; i < num_references; i++) {
        char *sequence = reference_seqs[i];
        reference_length = strlen(sequence);

        int previous_k = 0;
        int temp = 0;

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

            int k;
            for (k = 0; k < extended_chunk_size; k++) {
                ref_toparti[k] = sequence[previous_k + k];
            }
            ref_toparti[k] = '\0';

            // print to terminal
            printf("Chunk from sequence %d: %s\n", temp, ref_toparti);
            temp++;
            // write to output FASTA
            fprintf(output_file, ">%d_%d\n", temp, previous_k);
            write_fasta(output_file, ref_toparti, 60);

            previous_k += p;
        }
    }

    fclose(output_file);
    printf("Partitioned FASTA file written.\n");

    // free reference_seqs if needed (depends on how parse_fasta_file works)
    // free_reference_seqs(reference_seqs, num_references);

    return 0;
}
