#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

#include <cassert>
#include <vector>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/seqcommit/base/commitment.h"

namespace sxt::sqcnv {

static constexpr uint64_t block_size = 64;

//--------------------------------------------------------------------------------------------------
// get_value_sequence_size
//--------------------------------------------------------------------------------------------------
static void get_value_sequence_size(
    uint64_t &sequence_size, uint64_t &total_num_blocks, uint64_t &biggest_n,
    basct::cspan<mtxb::exponent_sequence> value_sequences) {
  biggest_n = 0;
  sequence_size = 0;
  total_num_blocks = 0;

  for (size_t c = 0; c < value_sequences.size(); ++c) {
    auto &data_commitment = value_sequences[c];

    assert(data_commitment.element_nbytes > 0 && data_commitment.element_nbytes <= 32);

    biggest_n = max(biggest_n, data_commitment.n);
    total_num_blocks += basn::divide_up(data_commitment.n, block_size);
    sequence_size += data_commitment.n * data_commitment.element_nbytes;
  }
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_kernel
//--------------------------------------------------------------------------------------------------
__global__ static void compute_commitments_kernel(
    c21t::element_p3 *partial_commitments,
    mtxb::exponent_sequence value_sequence) {
  extern __shared__ c21t::element_p3 reduction[];

  int tid = threadIdx.x;
  int n = value_sequence.n;
  int row_i = threadIdx.x + blockIdx.x * blockDim.x;

  partial_commitments[blockIdx.x] = reduction[tid] = c21cn::zero_p3_v;

  if (row_i >= n) return;

  uint8_t element_nbytes = value_sequence.element_nbytes;

  c21t::element_p3 g_i;

  sqcgn::compute_base_element(g_i, row_i);

  basct::cspan<uint8_t> exponent{value_sequence.data + row_i * element_nbytes,
                                 element_nbytes};
  c21o::scalar_multiply(reduction[tid], exponent, g_i);  // h_i = a_i * g_i

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
__global__ static void commitment_reduction_kernel(
    sqcb::commitment *final_commitment, c21t::element_p3 *partial_commitments,
    uint64_t n) {
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
// launch_commitment_kernels
//--------------------------------------------------------------------------------------------------
static void launch_commitment_kernels(
    memmg::managed_array<sqcb::commitment> &commitments_device,
    basct::cspan<mtxb::exponent_sequence> value_sequences) {
  uint64_t num_commitments = commitments_device.size();
  uint64_t sequence_size, total_num_blocks, biggest_n;

  get_value_sequence_size(sequence_size, total_num_blocks, biggest_n,
                          value_sequences);

  if (sequence_size == 0) return;

  const int num_streams = min(10000, (int)num_commitments);
  std::vector<basdv::stream> streams(num_streams);

  memmg::managed_array<uint8_t> data_table_device(sequence_size,
                                                  memr::get_device_resource());
  memmg::managed_array<c21t::element_p3> partial_commitments_device(
      total_num_blocks, memr::get_device_resource());

  uint8_t *data_table_device_ptr = data_table_device.data();
  c21t::element_p3 *partial_commitments_device_ptr =
      partial_commitments_device.data();

  for (int commitment_index = 0; commitment_index < num_commitments;
       ++commitment_index) {
    auto data_commitment = value_sequences[commitment_index];

    if (data_commitment.n > 0) {
      uint64_t commitment_index_size =
        data_commitment.n * data_commitment.element_nbytes;

      uint64_t shared_mem_size = block_size * sizeof(c21t::element_p3);

      uint64_t num_blocks =
          basn::divide_up(data_commitment.n, block_size);
          
      int stream_index =
          static_cast<int>((commitment_index / static_cast<float>(num_commitments)) * num_streams);

      auto curr_stream = streams[stream_index].raw_stream();
      
      basdv::async_memcpy_host_to_device(data_table_device_ptr,
                                        data_commitment.data,
                                        commitment_index_size, curr_stream);

      data_commitment.data = data_table_device_ptr;

      compute_commitments_kernel<<<num_blocks, block_size, shared_mem_size,
                                  curr_stream>>>(partial_commitments_device_ptr,
                                                  data_commitment);

      commitment_reduction_kernel<<<1, block_size, shared_mem_size,
                                    curr_stream>>>(
          &commitments_device[commitment_index], partial_commitments_device_ptr,
          static_cast<int>(num_blocks));

      data_table_device_ptr += commitment_index_size;
      partial_commitments_device_ptr += num_blocks;
    }
  }

  // synchronize all streams
  cudaDeviceSynchronize();
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_gpu
//--------------------------------------------------------------------------------------------------
void compute_commitments_gpu(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept {
  assert(commitments.size() == value_sequences.size());

  memmg::managed_array<sqcb::commitment> commitments_device(
       commitments.size(), memr::get_device_resource());

  launch_commitment_kernels(commitments_device, value_sequences);

  basdv::memcpy_device_to_host(commitments.data(), commitments_device.data(),
                             commitments_device.num_bytes());
}

}  // namespace sxt::sqcnv
