#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_SEQUENCES 100
#define MAX_SEQ_LEN 1073741824 // 10MB per sequence
#define BUFFER_SIZE 65536

#define QUERY_FOLDER "/home/dlsu-cse/Downloads/Testing2026/Testing/quetry"
#define REFERENCE_FOLDER "/home/dlsu-cse/Downloads/Testing2026/Testing/reftry"
#define RESULTS_FOLDER   "results"
#define RESULTS_CSV_FILE "results/FPGAresults2.csv"

typedef struct
{
    char name[256];
    unsigned char *data;
    long length;
} Sequence;

typedef struct
{
    Sequence seqs[MAX_SEQUENCES];
    int count;
} SequenceList;

// ============================================================================
// RESULT STRUCTURE
// ============================================================================

typedef struct {
    double hw_time_ms;
    int    final_score;          // Last Score (score at end of reference)
    int    lowest_score;         // Minimum edit distance found
    int   *lowest_indexes;       // Positions where lowest score occurs
    int    num_lowest_indexes;
    int    valid;                // 1 on success, 0 on failure
} PairResult;

/**
 * @brief Parse a Python-style index list: "[123, 456, 789]"
 * @return Heap-allocated int array; caller must free. *count_out set to size.
 */
static int* parse_index_list(const char* str, int* count_out)
{
    *count_out = 0;
    const char* start = strchr(str, '[');
    const char* end   = strchr(str, ']');
    if (!start || !end || end <= start) return NULL;

    // Estimate capacity by counting commas
    int capacity = 1;
    for (const char* p = start; p < end; p++) if (*p == ',') capacity++;

    int* arr = (int*)malloc(capacity * sizeof(int));
    int  n   = 0;
    const char* p = start + 1;
    while (p < end) {
        while (p < end && (*p == ' ' || *p == ',')) p++;
        if (p >= end) break;
        char* endptr;
        long val = strtol(p, &endptr, 10);
        if (endptr == p) break;
        arr[n++] = (int)val;
        p = endptr;
    }
    *count_out = n;
    return arr;
}

/* qsort comparator for string pointers */
static int cmp_str(const void* a, const void* b)
{
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

/**
 * @brief Scan a directory and return a sorted list of regular filenames.
 *        Uses a single readdir pass with dynamic growth — safe for large folders.
 * @param folder     Directory path.
 * @param count_out  Output: number of files found.
 * @return Heap-allocated array of heap-allocated filename strings; caller frees.
 */
static char** get_sorted_files(const char* folder, int* count_out)
{
    *count_out = 0;
    DIR* dir = opendir(folder);
    if (!dir) { fprintf(stderr, "Cannot open folder: %s\n", folder); return NULL; }

    int      capacity = 64;
    int      n        = 0;
    char**   files    = (char**)malloc((size_t)capacity * sizeof(char*));

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) continue;
        if (n == capacity) {
            capacity *= 2;
            files = (char**)realloc(files, (size_t)capacity * sizeof(char*));
        }
        files[n++] = strdup(entry->d_name);
    }
    closedir(dir);

    if (n == 0) { free(files); return NULL; }

    qsort(files, (size_t)n, sizeof(char*), cmp_str);

    *count_out = n;
    return files;
}

// Parse FASTA file
SequenceList *parse_fasta(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    SequenceList *list = (SequenceList *)malloc(sizeof(SequenceList));
    list->count = 0;

    char *line_buffer = (char *)malloc(MAX_SEQ_LEN); // For reading lines
    char *seq_buffer = (char *)malloc(MAX_SEQ_LEN);  // For accumulating sequence
    long seq_len = 0;
    char header[256];
    int in_header = 0;

    while (fgets(line_buffer, MAX_SEQ_LEN, fp))
    {
        char *line = line_buffer;

        // Remove newline (handles both \r\n and \n)
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        {
            line[len - 1] = '\0';
            len--;
        }

        if (line[0] == '>')
        {
            // Save previous sequence if exists
            if (seq_len > 0 && list->count < MAX_SEQUENCES)
            {
                Sequence *seq = &list->seqs[list->count];
                seq->data = (unsigned char *)malloc(seq_len);
                memcpy(seq->data, seq_buffer, seq_len);
                seq->length = seq_len;
                list->count++;
                seq_len = 0;
            }

            // Start new sequence
            strncpy(header, line + 1, sizeof(header) - 1);
            header[sizeof(header) - 1] = '\0';
            strcpy(list->seqs[list->count].name, header);
            in_header = 1;
        }
        else if (in_header && strlen(line) > 0)
        {
            // Accumulate sequence data
            strcpy(seq_buffer + seq_len, line);
            seq_len += strlen(line);
        }
    }

    // Save last sequence
    if (seq_len > 0 && list->count < MAX_SEQUENCES)
    {
        Sequence *seq = &list->seqs[list->count];
        seq->data = (unsigned char *)malloc(seq_len);
        memcpy(seq->data, seq_buffer, seq_len);
        seq->length = seq_len;
        printf("[PARSE] Sequence '%s': %ld bytes\n", seq->name, seq->length);
        list->count++;
    }

    free(line_buffer);
    free(seq_buffer);
    fclose(fp);

    printf("Parsed %d sequences from %s\n", list->count, filename);
    return list;
}

// Connect to FPGA server
int connect_to_server(const char *server_ip, int port)
{
    int sock;
    struct sockaddr_in server_addr;

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0)
    {
        printf("Socket creation failed! Error: %s\n", strerror(errno));
        return -1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0)
    {
        printf("Invalid address! Error: %s\n", strerror(errno));
        close(sock);
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        printf("Connection failed! Error: %s\n", strerror(errno));
        close(sock);
        return -1;
    }

    printf("Connected to %s:%d\n", server_ip, port);
    return sock;
}

// Send query and reference to FPGA; returns a PairResult with all parsed fields.
static PairResult send_to_fpga(int sock, Sequence *query, Sequence *ref)
{
    PairResult result = {0};
    result.valid       = 0;
    result.final_score = -1;
    result.lowest_score = -1;

    printf("[DEBUG] Query bytes before sending: %ld\n", query->length);
    printf("[DEBUG] Ref bytes before sending: %ld\n", ref->length);

    unsigned char header[8];
    int q_len = (int)query->length;
    int r_len = (int)ref->length;

    // Pack header: query_length, ref_length (4 bytes each)
    memcpy(header, &q_len, 4);
    memcpy(header + 4, &r_len, 4);

    if (send(sock, (const char *)header, 8, 0) < 0) {
        printf("Failed to send header! Error: %s\n", strerror(errno));
        return result;
    }

    // Send query
    int total_sent = 0;
    while (total_sent < q_len) {
        ssize_t sent = send(sock, (const char *)(query->data + total_sent),
                            q_len - total_sent, 0);
        if (sent < 0) { printf("Failed to send query! Error: %s\n", strerror(errno)); return result; }
        total_sent += sent;
    }

    // Send reference
    total_sent = 0;
    while (total_sent < r_len) {
        ssize_t sent = send(sock, (const char *)(ref->data + total_sent),
                            r_len - total_sent, 0);
        if (sent < 0) { printf("Failed to send reference! Error: %s\n", strerror(errno)); return result; }
        total_sent += sent;
    }

    printf("Sent: %s (%ld bytes) + %s (%ld bytes)\n",
           query->name, query->length, ref->name, ref->length);

    // Receive response
    char response[10240];
    ssize_t recv_len = recv(sock, response, sizeof(response) - 1, 0);
    if (recv_len < 0) {
        printf("Failed to receive response! Error: %s\n", strerror(errno));
        return result;
    }
    response[recv_len] = '\0';
    printf("Response: %s\n", response);

    // Parse response line by line
    char resp_copy[10240];
    strncpy(resp_copy, response, sizeof(resp_copy) - 1);
    resp_copy[sizeof(resp_copy) - 1] = '\0';

    char* line = strtok(resp_copy, "\n");
    while (line) {
        if (strstr(line, "Hardware Time:"))
            sscanf(line, "Hardware Time: %lf ms", &result.hw_time_ms);
        else if (strstr(line, "Final Score:"))
            sscanf(line, "Final Score: %d", &result.final_score);
        else if (strstr(line, "Lowest Score:"))
            sscanf(line, "Lowest Score: %d", &result.lowest_score);
        else if (strstr(line, "Indices:")) {
            char* list_start = strchr(line, '[');
            if (list_start)
                result.lowest_indexes = parse_index_list(list_start,
                                                         &result.num_lowest_indexes);
        }
        line = strtok(NULL, "\n");
    }

    result.valid = 1;
    return result;
}

/**
 * @brief Format an int array as a comma-separated string.
 *        Returns heap-allocated string; caller must free.
 *        Returns "N/A" when count == 0.
 */
static char* format_indices(int* indices, int count)
{
    if (count == 0 || indices == NULL) return strdup("N/A");
    char* result = (char*)malloc((size_t)count * 12 + 1);
    result[0] = '\0';
    for (int i = 0; i < count; i++) {
        char buf[20];
        sprintf(buf, "%d", indices[i]);
        strcat(result, buf);
        if (i < count - 1) strcat(result, ",");
    }
    return result;
}

int main(int argc, char *argv[])
{
    // Optional overrides: ./fasta_to_fpga [server_ip] [port]
    const char *server_ip = "192.168.2.99";
    int port = 5000;
    if (argc > 1) server_ip = argv[1];
    if (argc > 2) port = atoi(argv[2]);

    printf("=== FASTA to FPGA Client ===\n");
    printf("Query folder:  %s\n", QUERY_FOLDER);
    printf("Ref folder:    %s\n", REFERENCE_FOLDER);
    printf("Server:        %s:%d\n\n", server_ip, port);

    // ========== Scan input folders ==========
    int num_query_files = 0, num_ref_files = 0;
    char** query_files = get_sorted_files(QUERY_FOLDER, &num_query_files);
    char** ref_files   = get_sorted_files(REFERENCE_FOLDER, &num_ref_files);

    if (!query_files || num_query_files == 0) {
        fprintf(stderr, "No query files found in %s\n", QUERY_FOLDER);
        return 1;
    }
    if (!ref_files || num_ref_files == 0) {
        fprintf(stderr, "No reference files found in %s\n", REFERENCE_FOLDER);
        return 1;
    }

    printf("Found %d query file(s) and %d reference file(s) -> %d pair(s)\n",
           num_query_files, num_ref_files, num_query_files * num_ref_files);

    // ========== Create results folder and open CSV ==========
    struct stat st = {0};
    if (stat(RESULTS_FOLDER, &st) == -1) mkdir(RESULTS_FOLDER, 0755);

    FILE* csv = fopen(RESULTS_CSV_FILE, "w");
    if (!csv) { perror("Failed to create CSV file"); return 1; }

    fprintf(csv, "FPGA Run\n");
    fprintf(csv, "Query File,Query Length,Reference File,Reference Length,"
                 "Number of Hits,Hit Indexes,Lowest Score,Lowest Score Indexes,Last Score\n");

    int    pair_count    = 0;
    double total_hw_time = 0.0;

    // ========== Full cross-product: every query file x every reference file ==========
    for (int qf = 0; qf < num_query_files; qf++) {
        char query_path[1024];
        snprintf(query_path, sizeof(query_path), "%s/%s", QUERY_FOLDER, query_files[qf]);

        SequenceList* queries = parse_fasta(query_path);
        if (!queries || queries->count == 0) {
            fprintf(stderr, "Failed to load query: %s\n", query_files[qf]);
            if (queries) free(queries);
            continue;
        }
        Sequence* query    = &queries->seqs[0];   // 1 query per file
        int       query_len = (int)query->length;

        for (int rf = 0; rf < num_ref_files; rf++) {
            char ref_path[1024];
            snprintf(ref_path, sizeof(ref_path), "%s/%s", REFERENCE_FOLDER, ref_files[rf]);

            SequenceList* refs = parse_fasta(ref_path);
            if (!refs || refs->count == 0) {
                fprintf(stderr, "Failed to load reference: %s\n", ref_files[rf]);
                if (refs) free(refs);
                continue;
            }
            Sequence* ref    = &refs->seqs[0];    // 1 reference per file
            int       ref_len = (int)ref->length;

            printf("\n[Pair %d] %s  vs  %s\n", ++pair_count,
                   query_files[qf], ref_files[rf]);

            // Connect per pair (server expects one request per connection)
            int sock = connect_to_server(server_ip, port);
            if (sock < 0) {
                fprintf(stderr, "Failed to connect for pair %d\n", pair_count);
                for (int i = 0; i < refs->count; i++) free(refs->seqs[i].data);
                free(refs);
                continue;
            }

            PairResult res = send_to_fpga(sock, query, ref);
            close(sock);

            if (res.valid) {
                total_hw_time += res.hw_time_ms;

                // Hits = positions where edit distance == 0
                int  hit_count    = (res.lowest_score == 0) ? res.num_lowest_indexes : 0;
                int* hit_indexes  = (res.lowest_score == 0) ? res.lowest_indexes     : NULL;

                char* hit_idx_str = format_indices(hit_indexes, hit_count);
                char* low_idx_str = (hit_count > 0)
                                    ? strdup("N/A")
                                    : format_indices(res.lowest_indexes, res.num_lowest_indexes);

                char lowest_score_str[20];
                if (hit_count > 0) strcpy(lowest_score_str, "N/A");
                else               sprintf(lowest_score_str, "%d", res.lowest_score);

                // Console output
                printf("Number of Hits: %d\n",       hit_count);
                printf("Hit Indexes: %s\n",           hit_idx_str);
                printf("Lowest Score: %s\n",          lowest_score_str);
                printf("Lowest Score Indexes: %s\n",  low_idx_str);
                printf("Last Score: %d\n",            res.final_score);
                printf("Hardware Time: %.2f ms\n",    res.hw_time_ms);

                // CSV row (index fields quoted to handle commas inside them)
                fprintf(csv, "%s,%d,%s,%d,%d,\"%s\",%s,\"%s\",%d\n",
                        query_files[qf], query_len,
                        ref_files[rf],   ref_len,
                        hit_count,       hit_idx_str,
                        lowest_score_str, low_idx_str,
                        res.final_score);

                free(hit_idx_str);
                free(low_idx_str);
                if (res.lowest_indexes) free(res.lowest_indexes);
            } else {
                fprintf(stderr, "[Pair %d] FPGA communication failed, skipping CSV row\n",
                        pair_count);
            }

            for (int i = 0; i < refs->count; i++) free(refs->seqs[i].data);
            free(refs);
        }

        for (int i = 0; i < queries->count; i++) free(queries->seqs[i].data);
        free(queries);
    }

    fclose(csv);

    printf("\n=== Completed %d pairs ===\n", pair_count);
    printf("Total Hardware Time: %.2f ms\n", total_hw_time);
    printf("Results saved to %s\n", RESULTS_CSV_FILE);

    // Cleanup file lists
    for (int i = 0; i < num_query_files; i++) free(query_files[i]);
    free(query_files);
    for (int i = 0; i < num_ref_files; i++) free(ref_files[i]);
    free(ref_files);

    return 0;
}