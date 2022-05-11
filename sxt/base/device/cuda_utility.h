#pragma once

#include <iostream>
#include <cuda_runtime.h>

namespace sxt::basdv {

//--------------------------------------------------------------------------------------------------
// synchronize_all
//--------------------------------------------------------------------------------------------------
void synchronize_all() noexcept {
    cudaDeviceSynchronize();
}

//--------------------------------------------------------------------------------------------------
// handle_cuda_error
//--------------------------------------------------------------------------------------------------
void handle_cuda_error(cudaError_t cuda_err) noexcept {
    if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA ERROR while executing the kernel: " << std::string(cudaGetErrorString(cuda_err)) << std::endl;
        std::abort();
    }
}

//--------------------------------------------------------------------------------------------------
// get_num_blocks
//--------------------------------------------------------------------------------------------------
uint64_t get_num_blocks(int block_size, uint64_t size) noexcept {
    return (size + block_size - 1) / block_size;
}

}  // namespace sxt::basdv