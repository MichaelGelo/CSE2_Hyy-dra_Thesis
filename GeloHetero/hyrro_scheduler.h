/**
 * @file hyrro_scheduler.h
 * @brief Adaptive scheduler for heterogeneous GPU-FPGA workload distribution
 * 
 * This file implements an adaptive scheduling algorithm that dynamically
 * adjusts the work distribution between GPU and FPGA based on their
 * relative performance on previous tasks.
 * 
 * Key features:
 * - Multiplicative correction based on execution times
 * - Smoothing to prevent oscillation
 * - Thread-safe ratio updates
 */

#ifndef HYRRO_SCHEDULER_H
#define HYRRO_SCHEDULER_H

#include <pthread.h>
#include <math.h>
#include "config.h"

// ============================================================================
// SYNCHRONIZATION PRIMITIVES
// ============================================================================

/**
 * Mutex for protecting ratio calculations
 */
static pthread_mutex_t ratio_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * Condition variable for ratio thread handshake
 */
static pthread_cond_t ratio_cond = PTHREAD_COND_INITIALIZER;

// ============================================================================
// RATIO THREAD ARGUMENTS
// ============================================================================

/**
 * @brief Arguments for ratio calculation thread
 * 
 * This structure is used to pass timing data to the ratio calculation
 * thread and receive the updated ratio back.
 */
typedef struct {
    double prev_gpu_time;    ///< GPU execution time from previous iteration
    double prev_fpga_time;   ///< FPGA execution time from previous iteration
    float old_gpu_ratio;     ///< Current GPU work ratio (input)
    float new_gpu_ratio;     ///< Updated GPU work ratio (output)
    int taken;               ///< Handshake flag: 1 when args captured
} RatioThreadArgs;

// ============================================================================
// ADAPTIVE SCHEDULING ALGORITHM
// ============================================================================

/**
 * @brief Calculate new GPU ratio based on execution times
 * 
 * Uses multiplicative correction with smoothing to adjust work distribution.
 * The goal is to balance execution times: T_gpu ≈ T_fpga
 * 
 * Algorithm:
 * 1. Calculate correction factor: sqrt(fpga_time / gpu_time)
 * 2. Apply multiplicative update: proposed = old_ratio * factor
 * 3. Smooth with exponential moving average
 * 4. Clamp to valid range [MIN_GPU_RATIO, MAX_GPU_RATIO]
 * 
 * Intuition:
 * - If GPU is faster (gpu_time < fpga_time): increase GPU ratio
 * - If FPGA is faster (fpga_time < gpu_time): decrease GPU ratio
 * - sqrt() provides stable convergence
 * 
 * @param gpu_time GPU execution time in seconds
 * @param fpga_time FPGA execution time in seconds
 * @param old_ratio Current GPU ratio (0.0 to 1.0)
 * @return Updated GPU ratio
 */
static float calculate_gpu_ratio(double gpu_time, double fpga_time, float old_ratio) {
    // Validate old ratio
    if (old_ratio <= 0.0f) {
        old_ratio = GPU_SPEED_RATIO;  // Use default if invalid
    }
    
    // If either time is invalid, keep current ratio
    if (gpu_time <= 0.0 || fpga_time <= 0.0) {
        return old_ratio;
    }

    // === Multiplicative correction ===
    // Factor > 1: GPU should do more work (GPU is faster)
    // Factor < 1: GPU should do less work (FPGA is faster)
    double factor = sqrt(fpga_time / gpu_time);
    double proposed = (double)old_ratio * factor;

    // === Smoothing to prevent oscillation ===
    // Exponential moving average:
    // blended = (1 - α) * old + α * proposed
    // α = 0: no change (very stable, very slow)
    // α = 1: immediate change (fast adaptation, may oscillate)
    double blended = (1.0 - RATIO_SMOOTHING_ALPHA) * old_ratio + 
                     RATIO_SMOOTHING_ALPHA * proposed;

    // === Clamp to valid range ===
    if (blended < MIN_GPU_RATIO) {
        blended = MIN_GPU_RATIO;
    }
    if (blended > MAX_GPU_RATIO) {
        blended = MAX_GPU_RATIO;
    }

    return (float)blended;
}

/**
 * @brief Ratio calculation thread function
 * 
 * This function runs in a separate thread to compute the new work ratio
 * without blocking the main computation flow.
 * 
 * Thread lifecycle:
 * 1. Main thread creates thread with RatioThreadArgs
 * 2. This function locks mutex and captures timing data
 * 3. Sets 'taken' flag and signals main thread (handshake)
 * 4. Releases mutex and performs calculation
 * 5. Writes result to new_gpu_ratio
 * 6. Thread exits
 * 
 * @param arg Pointer to RatioThreadArgs structure
 * @return NULL
 */
void* ratio_thread_func(void* arg) {
    RatioThreadArgs* args = (RatioThreadArgs*)arg;

    // ========== CRITICAL SECTION: Capture shared data ==========
    pthread_mutex_lock(&ratio_mutex);

    double gpu_t = args->prev_gpu_time;
    double fpga_t = args->prev_fpga_time;
    float old_r = args->old_gpu_ratio;

    // Signal main thread that we've captured the data
    args->taken = 1;
    pthread_cond_signal(&ratio_cond);
    
    pthread_mutex_unlock(&ratio_mutex);
    // ========== END CRITICAL SECTION ==========

    // Compute new ratio (outside critical section)
    args->new_gpu_ratio = calculate_gpu_ratio(gpu_t, fpga_t, old_r);

    // Diagnostic output
    printf("[Scheduler] Times: GPU=%.4fs, FPGA=%.4fs | Ratio: %.3f→%.3f\n", 
           gpu_t, fpga_t, old_r, args->new_gpu_ratio);

    return NULL;
}

// ============================================================================
// HIGH-LEVEL RATIO UPDATE FUNCTION
// ============================================================================

/**
 * @brief Update work ratio based on execution times (blocking)
 * 
 * This is a synchronous wrapper around the ratio calculation.
 * Useful when you need the new ratio immediately.
 * 
 * @param gpu_time Recent GPU execution time
 * @param fpga_time Recent FPGA execution time
 * @param current_ratio Current GPU work ratio
 * @return Updated GPU work ratio
 */
static inline float update_ratio_sync(double gpu_time, double fpga_time, float current_ratio) {
    return calculate_gpu_ratio(gpu_time, fpga_time, current_ratio);
}

/**
 * @brief Start asynchronous ratio calculation thread
 * 
 * Non-blocking version that performs calculation in background.
 * Call wait_for_ratio_thread() to get result.
 * 
 * @param thread Output: pthread handle
 * @param args Ratio thread arguments (must remain valid until thread completes)
 * @param gpu_time Recent GPU execution time
 * @param fpga_time Recent FPGA execution time
 * @param current_ratio Current GPU work ratio
 * @return 0 on success, error code on failure
 */
static inline int start_ratio_thread(
    pthread_t* thread,
    RatioThreadArgs* args,
    double gpu_time,
    double fpga_time,
    float current_ratio)
{
    // Initialize arguments
    args->prev_gpu_time = gpu_time;
    args->prev_fpga_time = fpga_time;
    args->old_gpu_ratio = current_ratio;
    args->taken = 0;
    args->new_gpu_ratio = current_ratio;  // Default in case thread fails

    // Create thread
    int ret = pthread_create(thread, NULL, ratio_thread_func, args);
    if (ret != 0) {
        fprintf(stderr, "ERROR: failed to create ratio thread\n");
        return ret;
    }

    // Wait for handshake (thread has captured data)
    pthread_mutex_lock(&ratio_mutex);
    while (args->taken == 0) {
        pthread_cond_wait(&ratio_cond, &ratio_mutex);
    }
    pthread_mutex_unlock(&ratio_mutex);

    return 0;
}

/**
 * @brief Wait for ratio calculation thread to complete
 * 
 * @param thread pthread handle
 * @param args Ratio thread arguments
 * @return Updated GPU work ratio
 */
static inline float wait_for_ratio_thread(pthread_t thread, RatioThreadArgs* args) {
    pthread_join(thread, NULL);
    return args->new_gpu_ratio;
}

#endif // HYRRO_SCHEDULER_H