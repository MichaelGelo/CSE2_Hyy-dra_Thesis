// udp_receiver.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define FPGA_PORT     5005
#define CHUNK_SIZE    8192
#define MAX_FILENAME  256
#define SAVE_DIR      "/home/xilinx/"  // Change to your FPGA path

int main() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); return 1; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(FPGA_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(sockfd); return 1;
    }

    printf("UDP receiver listening on port %d...\n", FPGA_PORT);

    unsigned char buffer[CHUNK_SIZE];
    while (1) {
        // Receive header first
        socklen_t addr_len = sizeof(addr);
        ssize_t header_len = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                                      (struct sockaddr*)&addr, &addr_len);
        if (header_len <= 0) continue;
        buffer[header_len] = 0;

        // Parse header: "filename\nfilesize\n"
        char filename[MAX_FILENAME];
        long file_size;
        if (sscanf((char*)buffer, "%255[^\n]\n%ld\n", filename, &file_size) != 2) {
            fprintf(stderr, "Invalid header, skipping\n");
            continue;
        }

        // Open file to write
        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s%s", SAVE_DIR, filename);
        FILE *fp = fopen(fullpath, "wb");
        if (!fp) { perror("fopen"); continue; }

        printf("Receiving file %s (%ld bytes)...\n", filename, file_size);

        long received_bytes = 0;
        while (received_bytes < file_size) {
            ssize_t n = recvfrom(sockfd, buffer, CHUNK_SIZE, 0,
                                 (struct sockaddr*)&addr, &addr_len);
            if (n <= 0) continue;
            fwrite(buffer, 1, n, fp);
            received_bytes += n;
            printf("\rReceived %ld / %ld bytes", received_bytes, file_size);
            fflush(stdout);
        }

        printf("\nFile %s received successfully!\n", filename);
        fclose(fp);
    }

    close(sockfd);
    return 0;
}
