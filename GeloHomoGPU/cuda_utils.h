// cuda_utils.h
// CUDA error checking and timing utilities

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// ============================================================================
// CUDA ERROR CHECKING MACRO
// Wraps CUDA API calls and checks for errors, printing location if error occurs
// ============================================================================
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================================================
// HIGH-PRECISION TIMING
// Returns current time in seconds with microsecond precision
// ============================================================================
static inline double getNowSeconds() {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER count;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

#endif // CUDA_UTILS_H