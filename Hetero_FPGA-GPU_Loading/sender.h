#ifndef SENDER_H
#define SENDER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/time.h>

// ================= CONFIGURATION =================
#define FPGA_IP "192.168.2.99"
#define FPGA_PORT 5000
#define RECV_BUF_SIZE 4096
// =================================================

typedef struct {
    double hw_exec_time_ms;
    int final_score;
    int lowest_score;
    unsigned long *lowest_indexes; 
    int num_lowest_indexes;
    double total_exec_ms;
} FPGAResult;

static double get_elapsed_time_ms(struct timeval start, struct timeval end) {
    return ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);
}

// Send all bytes
static int send_all(int sock, const void *data, size_t size) {
    const char *ptr = (const char *)data;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t sent = send(sock, ptr, remaining, 0);
        if (sent <= 0) return -1;
        ptr += sent;
        remaining -= sent;
    }
    return 0;
}

// Helper to parse Python list string: "[123, 456, 789]"
static void parse_python_list(char* list_str, FPGAResult* res) {
    // 1. Remove brackets
    char* start = strchr(list_str, '[');
    char* end = strchr(list_str, ']');
    
    if (!start || !end) return;
    
    *end = '\0'; // Remove closing bracket
    start++;     // Skip opening bracket
    
    // 2. Count items
    int count = 0;
    char* temp = strdup(start);
    char* token = strtok(temp, ", ");
    while(token) {
        if(strlen(token) > 0) count++;
        token = strtok(NULL, ", ");
    }
    free(temp);
    
    res->num_lowest_indexes = count;
    if (count == 0) return;

    // 3. Allocate and fill
    res->lowest_indexes = (unsigned long*)malloc(count * sizeof(unsigned long));
    int i = 0;
    token = strtok(start, ", ");
    while(token) {
        // Python indices are ints, we store as unsigned long
        res->lowest_indexes[i++] = strtoul(token, NULL, 10);
        token = strtok(NULL, ", ");
    }
}

// === RAM-TO-RAM TCP EXECUTION (LEGACY/TEXT MODE) ===
static FPGAResult run_fpga_tcp_ram(const char* ref_seq, const char* query_seq) {
    FPGAResult result = {0};
    result.final_score = -1;
    result.lowest_score = -1;
    
    int sock;
    struct sockaddr_in server_addr;
    struct timeval start, end;

    // 1. Setup Socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        return result;
    }

    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(FPGA_PORT);
    inet_pton(AF_INET, FPGA_IP, &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection to FPGA failed");
        close(sock);
        return result;
    }

    gettimeofday(&start, NULL);

    // --- PROTOCOL MATCHING YOUR PYTHON SERVER ---
    
    int r_len = (int)strlen(ref_seq);
    int q_len = (int)strlen(query_seq);
    
    // 1. Send Header: [QueryLen, RefLen] (Native Little Endian)
    // Note: Python reads 'ii' (native), so we send raw ints.
    // Note: Python reads Query Length first!
    int header[2] = { q_len, r_len };
    send_all(sock, header, sizeof(header));

    // 2. Send Payloads: Query First, then Reference
    send_all(sock, query_seq, q_len);
    send_all(sock, ref_seq, r_len);

    // 3. Receive Text Response
    char buffer[RECV_BUF_SIZE];
    memset(buffer, 0, RECV_BUF_SIZE);
    
    // Simple read (assuming response fits in one packet, which fits your text format)
    ssize_t n = recv(sock, buffer, RECV_BUF_SIZE - 1, 0);
    
    if (n > 0) {
        buffer[n] = '\0'; // Null terminate
        
        // Parse Text: "Final Score: 123"
        char* line = strtok(buffer, "\n");
        while(line) {
            if (strstr(line, "Hardware Time:")) {
                sscanf(line, "Hardware Time: %lf", &result.hw_exec_time_ms);
            }
            else if (strstr(line, "Final Score:")) {
                sscanf(line, "Final Score: %d", &result.final_score);
            }
            else if (strstr(line, "Lowest Score:")) {
                sscanf(line, "Lowest Score: %d", &result.lowest_score);
            }
            else if (strstr(line, "Indices:")) {
                // "Indices: [10, 20]"
                char* list_start = strchr(line, '[');
                if (list_start) {
                    parse_python_list(list_start, &result);
                }
            }
            line = strtok(NULL, "\n");
        }
    }

    gettimeofday(&end, NULL);
    result.total_exec_ms = get_elapsed_time_ms(start, end) / 1000.0;
    close(sock);
    return result;
}

#endif // SENDER_H