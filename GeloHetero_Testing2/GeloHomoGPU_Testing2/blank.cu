#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaFree(0);  // forces context creation
    std::cout << "Context created\n";
    getchar();
}