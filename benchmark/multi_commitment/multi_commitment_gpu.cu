#include "benchmark/multi_commitment/multi_commitment_gpu.h"

#include "sxt/base/container/span.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/seqcommit/naive/commitment_computation.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

#include <iostream>
#include <cstdio>
#include <string>

namespace sxt {

constexpr uint64_t block_size = 128;

//--------------------------------------------------------------------------------------------------
// get_num_blocks
//--------------------------------------------------------------------------------------------------
static uint64_t get_num_blocks(uint64_t size) {
    return (size + block_size - 1) / block_size;
}

//--------------------------------------------------------------------------------------------------
// handle_cuda_error
//--------------------------------------------------------------------------------------------------
static void handle_cuda_error(cudaError_t cuda_err) {
    if (cuda_err != cudaSuccess) {
        std::cout << "CUDA ERROR while executing the kernel: " << std::string(cudaGetErrorString(cuda_err)) << std::endl;
        std::abort();
    }
}

//--------------------------------------------------------------------------------------------------
// multi_commitment_kernel
//--------------------------------------------------------------------------------------------------
__global__ void multi_commitment_kernel(
    uint64_t num_rows, uint64_t num_cols, uint64_t element_nbytes, uint8_t *data_table, sqcb::commitment *commitment_results) {

    int curr_col = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (curr_col < num_cols) {
        sqcb::commitment curr_commitment;
        mtxb::exponent_sequence curr_data_col;

        curr_data_col.n = num_rows;
        curr_data_col.element_nbytes = element_nbytes;
        curr_data_col.data = data_table + curr_col * num_rows * element_nbytes;

        basct::span<sqcb::commitment> commitments(&curr_commitment, 1);
        basct::cspan<mtxb::exponent_sequence> value_sequences(&curr_data_col, 1);

        sqcnv::compute_commitments(commitments, value_sequences);

        commitment_results[curr_col] = curr_commitment;
    }
}

//--------------------------------------------------------------------------------------------------
// multi_commitment_gpu
//--------------------------------------------------------------------------------------------------
void multi_commitment_gpu(
    memmg::managed_array<sqcb::commitment> &commitments_per_col,
    uint64_t rows, uint64_t cols, uint64_t element_nbytes,
    const memmg::managed_array<uint8_t> &data_table_host) noexcept {

    uint64_t num_blocks = get_num_blocks(cols);

    memmg::managed_array<sqcb::commitment> commitment_results(cols, memr::get_device_resource());

    memmg::managed_array<uint8_t> data_table_device(data_table_host.size(), memr::get_device_resource());

    handle_cuda_error(cudaMemcpy(data_table_device.data(), data_table_host.data(), data_table_host.num_bytes(), cudaMemcpyHostToDevice));

    multi_commitment_kernel<<<num_blocks, block_size>>>(rows, cols, element_nbytes, data_table_device.data(), commitment_results.data());

    handle_cuda_error(cudaMemcpy(commitments_per_col.data(), commitment_results.data(), commitment_results.num_bytes(), cudaMemcpyDeviceToHost));
}
} // namespace sxt
