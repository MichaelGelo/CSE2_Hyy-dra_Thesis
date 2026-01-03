// C_utils.h
#ifdef __cplusplus
extern "C" {
#endif

char* read_file_into_string(const char* filename);
char** parse_fasta_file(const char *filename, int *num_sequences);

#ifdef __cplusplus
}
#endif