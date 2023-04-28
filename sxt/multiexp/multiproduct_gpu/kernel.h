#pragma once

#include <cassert>

#include "sxt/algorithm/base/gather_mapper.h"
#include "sxt/algorithm/base/reducer.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/execution/kernel/launch.h"
#include "sxt/multiexp/multiproduct_gpu/block_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// multiproduct_kernel
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer>
__global__ void multiproduct_kernel(typename Reducer::value_type* out,
                                    const typename Reducer::value_type* generators,
                                    const int* indexes,
                                    const block_computation_descriptor* block_descriptors) {
  using T = typename Reducer::value_type;
  extern __shared__ T shared_data[];
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto descriptor = block_descriptors[block_index];
  algb::gather_mapper<T, int> mapper{generators, indexes + descriptor.index_first};
  // Note: It's expected that most products will be of similar length and
  // hence share a common block size, but we allow for the same kernel to
  // compute product reductions with varying block sizes.
  xenk::launch_kernel(
      descriptor.block_size,
      [=]<unsigned BlockSize>(std::integral_constant<unsigned, BlockSize>) noexcept {
        assert(block_index >= descriptor.block_offset);
        auto index = (block_index - descriptor.block_offset) * (BlockSize * 2) + thread_index;
        auto step = BlockSize * 2 * descriptor.reduction_num_blocks;
        // If BlockSize is less than the maximum block size (the size the kernel was launched with),
        // then treat the some of the threads as inactive.
        if (thread_index < BlockSize) {
          algr::thread_reduce<Reducer, BlockSize>(out + block_index, shared_data, mapper,
                                                  descriptor.n, step, thread_index, index);
        }
      });
}
} // namespace sxt::mtxmpg
