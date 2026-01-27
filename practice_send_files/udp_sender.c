// udp_sender.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define FPGA_IP       "192.168.2.99"
#define FPGA_PORT     5005
#define CHUNK_SIZE    8192
#define NUM_FILES     2  // number of predefined files

// Predefined files to send
const char *files_to_send[NUM_FILES] = {
    "/home/jetson/fpga_ref0.fasta",
    "/home/jetson/fpga_ref1.fasta"
};

int send_file_udp(int sockfd, struct sockaddr_in *fpga_addr, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("fopen"); return -1; }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Send header: "basename\nfilesize\n"
    char header[1024];
    const char *basename = strrchr(filename, '/');
    basename = basename ? basename + 1 : filename;
    int header_len = snprintf(header, sizeof(header), "%s\n%ld\n", basename, file_size);

    if (sendto(sockfd, header, header_len, 0,
               (struct sockaddr*)fpga_addr, sizeof(*fpga_addr)) < 0) {
        perror("sendto header"); fclose(fp); return -1;
    }

    // Send file in chunks
    unsigned char buffer[CHUNK_SIZE];
    size_t bytes_read;
    long total_sent = 0;

    while ((bytes_read = fread(buffer, 1, CHUNK_SIZE, fp)) > 0) {
        ssize_t sent = sendto(sockfd, buffer, bytes_read, 0,
                              (struct sockaddr*)fpga_addr, sizeof(*fpga_addr));
        if (sent < 0) { perror("sendto"); fclose(fp); return -1; }
        total_sent += sent;
        printf("\rSent %ld / %ld bytes", total_sent, file_size);
        fflush(stdout);
    }

    printf("\nFile %s sent successfully!\n", basename);
    fclose(fp);
    return 0;
}

int main() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); return 1; }

    struct sockaddr_in fpga_addr;
    memset(&fpga_addr, 0, sizeof(fpga_addr));
    fpga_addr.sin_family = AF_INET;
    fpga_addr.sin_port = htons(FPGA_PORT);
    if (inet_pton(AF_INET, FPGA_IP, &fpga_addr.sin_addr) <= 0) {
        perror("inet_pton"); close(sockfd); return 1;
    }

    for (int i = 0; i < NUM_FILES; i++) {
        if (send_file_udp(sockfd, &fpga_addr, files_to_send[i]) != 0) {
            fprintf(stderr, "Failed to send file %s\n", files_to_send[i]);
        }
    }

    close(sockfd);
    return 0;
}
