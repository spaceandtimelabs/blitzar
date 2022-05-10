#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

#include <cassert>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/seqcommit/base/base_element.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/seqcommit/naive/reduce_exponent.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/curve21/constant/zero.h"

#include <iostream>

namespace sxt::sqcnv {

constexpr int block_size = 32;

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
        std::cerr << "CUDA ERROR while executing the kernel: " << std::string(cudaGetErrorString(cuda_err)) << std::endl;
        std::abort();
    }
}

//--------------------------------------------------------------------------------------------------
// get_value_sequence_size
//--------------------------------------------------------------------------------------------------
static void get_value_sequence_size(uint64_t &sequence_size, uint64_t &total_num_blocks, uint64_t &biggest_n, basct::cspan<mtxb::exponent_sequence> value_sequences) {
    biggest_n = 0;
    sequence_size = 0;
    total_num_blocks = 0;
    
    for (size_t c = 0; c < value_sequences.size(); ++c) {
        auto &data_col = value_sequences[c];

        assert(data_col.n != 0);
        assert(data_col.element_nbytes < 0 || data_col.element_nbytes > 32);

        biggest_n = max(biggest_n, data_col.n);
        total_num_blocks += get_num_blocks(data_col.n);
        sequence_size += data_col.n * data_col.element_nbytes;
    }
}

//--------------------------------------------------------------------------------------------------
// fill_data_gpu
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void fill_data_gpu(uint8_t a_i[32], const uint8_t *bytes_row_i_column_k, uint8_t size_row_data) noexcept {
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        // from (0) to (size_row_data - 1) we are populating a_i[j] with data values
        // padding zeros from (size_row_data) to (31)
        a_i[j] = (j < size_row_data) ? bytes_row_i_column_k[j] : 0;
    }

    if (a_i[31] > 127) {
        reduce_exponent(a_i); // a_i = a_i % (2^252 + 27742317777372353535851937790883648493)
    }
}

//--------------------------------------------------------------------------------------------------
// compute_col_commitments_kernel
//--------------------------------------------------------------------------------------------------
__global__ void compute_col_commitments_kernel(
    c21t::element_p3 *partial_commitments, mtxb::exponent_sequence value_sequence, c21t::element_p3 *random_buffer) {
    
    extern __shared__ c21t::element_p3 reduction[];

    int tid = threadIdx.x;
    int n = value_sequence.n;
    int row_i = threadIdx.x + blockIdx.x * blockDim.x;

    partial_commitments[blockIdx.x] = reduction[tid] = c21cn::zero_p3_v;

    if (row_i >= n) return;

    uint8_t element_nbytes = value_sequence.element_nbytes;

    uint8_t a_i[32];
    
    c21t::element_p3 g_i = random_buffer[row_i];

    // fill a_i, inserting data values at the beginning and padding zeros at the end of a_i
    fill_data_gpu(a_i, value_sequence.data + row_i * element_nbytes, element_nbytes);

    c21o::scalar_multiply(reduction[tid], a_i, g_i); // h_i = a_i * g_i

    __syncthreads();

    #pragma unroll
    for (int stride = block_size; stride > 1; stride >>= 1) {
        if (blockDim.x >= stride && tid < (stride >> 1) && row_i + stride / 2 < n) {
            c21o::add(reduction[tid], reduction[tid], reduction[tid + stride / 2]);
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial_commitments[blockIdx.x] = reduction[0];
    }
}

//--------------------------------------------------------------------------------------------------
// commitment_reduction_kernel
//--------------------------------------------------------------------------------------------------
__global__ void commitment_reduction_kernel(sqcb::commitment *final_commitment, c21t::element_p3 *partial_commitments, int n) {
    extern __shared__ c21t::element_p3 reduction[];

    int tid = threadIdx.x;

    reduction[tid] = c21cn::zero_p3_v;

    if (tid >= n) return;

    for (int i = tid; i < n; i += block_size) {
        c21o::add(reduction[tid], reduction[tid], partial_commitments[i]);
    }

    __syncthreads();

    if (tid == 0) {
        int reduction_size = min(n, block_size);
        
        for (int i = 1; i < reduction_size; ++i) {
            c21o::add(reduction[0], reduction[0], reduction[i]);
        }

        c21rs::to_bytes(final_commitment->data(), reduction[0]);
    }
}

//--------------------------------------------------------------------------------------------------
// generate_random_vals_kernel
//--------------------------------------------------------------------------------------------------
__global__ void generate_random_vals_kernel(uint64_t n, c21t::element_p3 *random_buffer) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        c21t::element_p3 g_i;

        sqcb::compute_base_element(g_i, idx);

        random_buffer[idx] = g_i;
    }
}

//--------------------------------------------------------------------------------------------------
// launch_kernels
//--------------------------------------------------------------------------------------------------
static void launch_kernels(
    memmg::managed_array<sqcb::commitment> &commitments_device,
    basct::cspan<mtxb::exponent_sequence> value_sequences) {
    
    uint64_t num_cols = commitments_device.size();
    uint64_t sequence_size, total_num_blocks, biggest_n;

    get_value_sequence_size(sequence_size, total_num_blocks, biggest_n, value_sequences);

    memmg::managed_array<cudaStream_t> streams(num_cols);
    memmg::managed_array<c21t::element_p3> random_values(biggest_n, memr::get_device_resource());
    memmg::managed_array<uint8_t> data_table_device(sequence_size, memr::get_device_resource());
    memmg::managed_array<c21t::element_p3> partial_commitments_device(total_num_blocks, memr::get_device_resource());

    generate_random_vals_kernel<<<get_num_blocks(biggest_n), block_size>>>(biggest_n, random_values.data());
    cudaDeviceSynchronize();

    uint8_t *data_table_device_ptr = data_table_device.data();
    c21t::element_p3 *partial_commitments_device_ptr = partial_commitments_device.data();

    for (int curr_col = 0; curr_col < num_cols; ++curr_col) {
        auto &curr_stream = streams[curr_col];

        handle_cuda_error(cudaStreamCreate(&curr_stream));
        
        mtxb::exponent_sequence data_col_device;
        auto data_col_host = value_sequences[curr_col];
        uint64_t curr_col_size = data_col_host.n * data_col_host.element_nbytes;

        data_col_device.n = data_col_host.n;
        data_col_device.element_nbytes = data_col_host.element_nbytes;
        data_col_device.data = data_table_device_ptr;

        handle_cuda_error(cudaMemcpyAsync(data_table_device_ptr, data_col_host.data, curr_col_size, cudaMemcpyHostToDevice, curr_stream));

        uint64_t num_blocks = get_num_blocks(data_col_device.n);
        uint64_t shared_mem_size = block_size * sizeof(c21t::element_p3);

        compute_col_commitments_kernel<<<num_blocks, block_size, shared_mem_size, curr_stream>>>(partial_commitments_device_ptr, data_col_device, random_values.data());

        commitment_reduction_kernel<<<1, block_size, shared_mem_size, curr_stream>>>(&commitments_device[curr_col], partial_commitments_device_ptr, (int) num_blocks);

        data_table_device_ptr += curr_col_size;
        partial_commitments_device_ptr += num_blocks;
    }

    // synchronize all kernels
    for (size_t c = 0; c < num_cols; ++c) {
        handle_cuda_error(cudaStreamSynchronize(streams[c]));
        handle_cuda_error(cudaStreamDestroy(streams[c]));
    }
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_gpu
//--------------------------------------------------------------------------------------------------
void compute_commitments_gpu(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept {
    
    assert(commitments.size() == value_sequences.size());

    uint64_t num_cols = commitments.size();

    memmg::managed_array<sqcb::commitment> commitments_device(num_cols, memr::get_device_resource());

    launch_kernels(commitments_device, value_sequences);

    handle_cuda_error(cudaMemcpy(commitments.data(), commitments_device.data(), commitments_device.num_bytes(), cudaMemcpyDeviceToHost));
}
} // namespace sxt
