#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

#include <vector>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/base/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::sqcnv {

static constexpr uint64_t block_size = 64;

//--------------------------------------------------------------------------------------------------
// get_value_sequence_size
//--------------------------------------------------------------------------------------------------
static void get_value_sequence_size(uint64_t& sequence_size, uint64_t& total_num_blocks,
                                    basct::cspan<mtxb::exponent_sequence> value_sequences,
                                    uint64_t generators_length) {
  sequence_size = 0;
  total_num_blocks = 0;

  for (size_t c = 0; c < value_sequences.size(); ++c) {
    auto& data_commitment = value_sequences[c];

    SXT_DEBUG_ASSERT(data_commitment.element_nbytes > 0 && data_commitment.element_nbytes <= 32);

    SXT_DEBUG_ASSERT(generators_length >= data_commitment.n);

    total_num_blocks += basn::divide_up(data_commitment.n, block_size);
    sequence_size += data_commitment.n * data_commitment.element_nbytes;
  }
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_kernel
//--------------------------------------------------------------------------------------------------
__global__ static void compute_commitments_kernel(c21t::element_p3* partial_commitments,
                                                  mtxb::exponent_sequence value_sequence,
                                                  c21t::element_p3* generators) {
  extern __shared__ c21t::element_p3 reduction[];

  int tid = threadIdx.x;
  int n = value_sequence.n;
  int row_i = threadIdx.x + blockIdx.x * blockDim.x;

  partial_commitments[blockIdx.x] = reduction[tid] = c21cn::zero_p3_v;

  if (row_i >= n)
    return;

  uint8_t element_nbytes = value_sequence.element_nbytes;

  basct::cspan<uint8_t> exponent{value_sequence.data + row_i * element_nbytes, element_nbytes};
  c21o::scalar_multiply(reduction[tid], exponent, generators[row_i]); // h_i = a_i * g_i

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
__global__ static void commitment_reduction_kernel(rstt::compressed_element* final_commitment,
                                                   c21t::element_p3* partial_commitments,
                                                   uint64_t n) {
  extern __shared__ c21t::element_p3 reduction[];

  int tid = threadIdx.x;

  reduction[tid] = c21cn::zero_p3_v;

  if (tid >= n)
    return;

  for (int i = tid; i < n; i += block_size) {
    c21o::add(reduction[tid], reduction[tid], partial_commitments[i]);
  }

  __syncthreads();

  if (tid == 0) {
    int reduction_size = min(n, block_size);

    for (int i = 1; i < reduction_size; ++i) {
      c21o::add(reduction[0], reduction[0], reduction[i]);
    }

    rstb::to_bytes(final_commitment->data(), reduction[0]);
  }
}

//--------------------------------------------------------------------------------------------------
// launch_commitment_kernels
//--------------------------------------------------------------------------------------------------
static void
launch_commitment_kernels(memmg::managed_array<rstt::compressed_element>& commitments_device,
                          basct::cspan<mtxb::exponent_sequence> value_sequences,
                          basct::cspan<c21t::element_p3> generators) {
  if (generators.size() == 0)
    return;

  uint64_t sequence_size, total_num_blocks;
  get_value_sequence_size(sequence_size, total_num_blocks, value_sequences, generators.size());

  const int num_streams = min(10000, static_cast<int>(commitments_device.size()));
  std::vector<xenb::stream> streams(num_streams);

  memmg::managed_array<uint8_t> data_table_device(sequence_size, memr::get_device_resource());
  memmg::managed_array<c21t::element_p3> partial_commitments_device(total_num_blocks,
                                                                    memr::get_device_resource());
  memmg::managed_array<c21t::element_p3> generators_device(generators.size(),
                                                           memr::get_managed_device_resource());

  basdv::memcpy_host_to_device(generators_device.data(), generators.data(),
                               generators.size() * sizeof(c21t::element_p3));

  uint8_t* data_table_dev_ptr = data_table_device.data();

  c21t::element_p3* partial_commits_dev_ptr = partial_commitments_device.data();

  // each cuda stream computes asynchronously one commitment
  for (int commit_index = 0; commit_index < commitments_device.size(); ++commit_index) {
    auto data_commit = value_sequences[commit_index];

    // we only process if there is at least one row. Otherwise, commitment is zero
    if (data_commit.n > 0) {
      uint64_t commit_index_size = data_commit.n * data_commit.element_nbytes;

      uint64_t shared_mem = block_size * sizeof(c21t::element_p3);

      uint64_t num_blocks = basn::divide_up(data_commit.n, block_size);

      int stream_index = static_cast<int>(
          (commit_index / static_cast<float>(commitments_device.size())) * num_streams);

      auto curr_stream = streams[stream_index].raw_stream();

      // We asynchronously move the current
      // sequence to the device memory, using
      // for that the current cuda stream.
      basdv::async_memcpy_host_to_device(data_table_dev_ptr, data_commit.data, commit_index_size,
                                         curr_stream);

      data_commit.data = data_table_dev_ptr;

      // We split the current sequence into multiple cuda blocks.
      // Each block is responsible for computing the
      // commitment related to the current sequence subset
      // that was assigned to the block. Because
      // the commitments of each block must be summed
      // together in a final single result, we add the
      // next cuda kernel. We use the current cuda stream
      // to process this kernel.
      compute_commitments_kernel<<<num_blocks, block_size, shared_mem, curr_stream>>>(
          partial_commits_dev_ptr, data_commit, generators_device.data());

      // This cuda kernel sums the commitments generated
      // by each cuda block from the previous execution,
      // storing the result commitment in the
      // `commitments_device[commit_index]`. We use the current cuda stream
      // to process this kernel.
      commitment_reduction_kernel<<<1, block_size, shared_mem, curr_stream>>>(
          &commitments_device[commit_index], partial_commits_dev_ptr, static_cast<int>(num_blocks));

      // We update the device pointers.
      // We use this scheme instead of
      // indexing, because each commitment
      // sequence can have a different number
      // of rows and data types
      data_table_dev_ptr += commit_index_size;
      partial_commits_dev_ptr += num_blocks;
    }
  }

  // synchronize all streams
  cudaDeviceSynchronize();
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_gpu
//--------------------------------------------------------------------------------------------------
void compute_commitments_gpu(basct::span<rstt::compressed_element> commitments,
                             basct::cspan<mtxb::exponent_sequence> value_sequences,
                             basct::cspan<c21t::element_p3> generators) noexcept {
  SXT_DEBUG_ASSERT(commitments.size() == value_sequences.size());

  // allocates memory to commitments in the device
  memmg::managed_array<rstt::compressed_element> commitments_device(commitments.size(),
                                                                    memr::get_device_resource());

  // we need to initialize all commitment sequence memory to zero,
  // given that `launch_commitment_kernels` will not launch kernels
  // for zero-length sequences
  basdv::memset_device(commitments_device.data(), 0,
                       commitments_device.size() * sizeof(rstt::compressed_element));

  // compute all the commitments
  // for each one of the sequences in
  // `value_sequences` span
  launch_commitment_kernels(commitments_device, value_sequences, generators);

  // bring the commitment results
  // from the device memory to the host
  basdv::memcpy_device_to_host(commitments.data(), commitments_device.data(),
                               commitments_device.num_bytes());
}

} // namespace sxt::sqcnv
