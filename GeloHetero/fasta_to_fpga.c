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

#define MAX_SEQUENCES 100
#define MAX_SEQ_LEN 10000000 // 10MB per sequence
#define BUFFER_SIZE 65536

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

// Send query and reference to FPGA
int send_to_fpga(int sock, Sequence *query, Sequence *ref, double *hw_time_acc)
{
    printf("[DEBUG] Query bytes before sending: %ld\n", query->length);
    printf("[DEBUG] Ref bytes before sending: %ld\n", ref->length);

    unsigned char header[8];
    int q_len = (int)query->length;
    int r_len = (int)ref->length;

    // Pack header: query_length, ref_length (as 4-byte ints each)
    memcpy(header, &q_len, 4);
    memcpy(header + 4, &r_len, 4);

    // Send header
    if (send(sock, (const char *)header, 8, 0) < 0)
    {
        printf("Failed to send header! Error: %s\n", strerror(errno));
        return -1;
    }

    // Send query
    int total_sent = 0;
    while (total_sent < q_len)
    {
        ssize_t sent = send(sock, (const char *)(query->data + total_sent),
                            q_len - total_sent, 0);
        if (sent < 0)
        {
            printf("Failed to send query! Error: %s\n", strerror(errno));
            return -1;
        }
        total_sent += sent;
    }

    // Send reference
    total_sent = 0;
    while (total_sent < r_len)
    {
        ssize_t sent = send(sock, (const char *)(ref->data + total_sent),
                            r_len - total_sent, 0);
        if (sent < 0)
        {
            printf("Failed to send reference! Error: %s\n", strerror(errno));
            return -1;
        }
        total_sent += sent;
    }

    printf("Sent: %s (%ld bytes) + %s (%ld bytes)\n",
           query->name, query->length, ref->name, ref->length);

    // Receive response
    char response[10240];
    ssize_t recv_len = recv(sock, response, sizeof(response) - 1, 0);
    if (recv_len < 0)
    {
        printf("Failed to receive response! Error: %s\n", strerror(errno));
        return -1;
    }

    response[recv_len] = '\0';
    printf("Response: %s\n", response);

    // Extract hardware time from response
    double hw_time = 0.0;
    char *time_str = strstr(response, "Hardware Time:");
    if (time_str && sscanf(time_str, "Hardware Time: %lf ms", &hw_time) == 1)
    {
        *hw_time_acc += hw_time;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    const char *query_file = "/home/dlsu-cse/Downloads/Stress Test/stestque_256.fasta";
    const char *ref_file = "/home/dlsu-cse/Downloads/Stress Test/stestm50ref_10M.fasta";
    const char *server_ip = "192.168.2.99"; // PYNQ board IP
    int port = 5000;

    // Parse command line arguments
    if (argc > 1)
        query_file = argv[1];
    if (argc > 2)
        ref_file = argv[2];
    if (argc > 3)
        server_ip = argv[3];
    if (argc > 4)
        port = atoi(argv[4]);

    printf("=== FASTA to FPGA Client ===\n");
    printf("Query file: %s\n", query_file);
    printf("Ref file: %s\n", ref_file);
    printf("Server: %s:%d\n\n", server_ip, port);

    // Parse FASTA files
    SequenceList *queries = parse_fasta(query_file);
    SequenceList *refs = parse_fasta(ref_file);

    if (!queries || !refs || queries->count == 0 || refs->count == 0)
    {
        printf("Error: Failed to parse FASTA files\n");
        return 1;
    }

    printf("Queries: %d, References: %d\n\n", queries->count, refs->count);

    // Send pairs in order: for each ref, send all queries
    int pair_count = 0;
    double total_hw_time = 0.0;
    for (int r = 0; r < refs->count; r++)
    {
        printf("\n--- Processing Reference %d: %s ---\n", r + 1, refs->seqs[r].name);

        for (int q = 0; q < queries->count; q++)
        {
            printf("[Pair %d] ", ++pair_count);

            // Connect per pair to match server behavior (one request per connection)
            int sock = connect_to_server(server_ip, port);
            if (sock < 0)
            {
                printf("Failed to connect to server\n");
                return 1;
            }

            if (send_to_fpga(sock, &queries->seqs[q], &refs->seqs[r], &total_hw_time) != 0)
            {
                printf("Error sending pair\n");
                close(sock);
                return 1;
            }

            close(sock);
        }
    }

    // Cleanup
    for (int i = 0; i < queries->count; i++)
    {
        free(queries->seqs[i].data);
    }
    free(queries);

    for (int i = 0; i < refs->count; i++)
    {
        free(refs->seqs[i].data);
    }
    free(refs);

    printf("\n=== Completed %d pairs ===\n", pair_count);
    printf("Total Hardware Execution Time: %.2f ms\n", total_hw_time);
    return 0;
}