#pragma once

#include <cassert>
#include <concepts>

#include "sxt/algorithm/base/mapper.h"
#include "sxt/algorithm/base/reducer.h"
#include "sxt/algorithm/base/reducer_utility.h"
#include "sxt/algorithm/reduction/warp_reduction.h"

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// thread_reduce
//--------------------------------------------------------------------------------------------------
/**
 * Perform the work in a parallel reduction for a single thread.
 *
 * This is based off of nvidia's guide
 *  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template <algb::reducer Reducer, unsigned int BlockSize, algb::mapper Mapper>
  requires std::same_as<typename Reducer::value_type, typename Mapper::value_type>
__device__ void thread_reduce(typename Reducer::value_type* out,
                              typename Reducer::value_type* shared_data, Mapper mapper,
                              unsigned int n, unsigned int step, unsigned int thread_index,
                              unsigned int index) {
  // clang-format off
  assert(
      (index + BlockSize < n || (BlockSize == 1 && n > 0)) &&
      "we assume the size of the reduction has been normalized"
  );
  // clang-format on

  // handle BlockSize == 1 as a special case
  if constexpr (BlockSize == 1) {
    if (n == 1) {
      mapper.map_index(*out, index);
      return;
    }
  }

  mapper.map_index(shared_data[thread_index], index);
  algb::accumulate<Reducer>(shared_data[thread_index], mapper, index + BlockSize);
  index += step;
  auto index_p = index + BlockSize;
  while (index_p < n) {
    algb::accumulate<Reducer>(shared_data[thread_index], mapper, index);
    algb::accumulate<Reducer>(shared_data[thread_index], mapper, index + BlockSize);
    index += step;
    index_p = index + BlockSize;
  }
  if (index < n) {
    algb::accumulate<Reducer>(shared_data[thread_index], mapper, index);
  }
  if constexpr (BlockSize > 32) {
    __syncthreads();
  }
  if constexpr (BlockSize >= 512) {
    if (thread_index < 256) {
      Reducer::accumulate(shared_data[thread_index], shared_data[thread_index + 256]);
    }
    __syncthreads();
  }
  if constexpr (BlockSize >= 256) {
    if (thread_index < 128) {
      Reducer::accumulate(shared_data[thread_index], shared_data[thread_index + 128]);
    }
    __syncthreads();
  }
  if constexpr (BlockSize >= 128) {
    if (thread_index < 64) {
      Reducer::accumulate(shared_data[thread_index], shared_data[thread_index + 64]);
    }
    __syncthreads();
  }
  if (thread_index < 32) {
    warp_reduce<Reducer, BlockSize>(shared_data, thread_index);
  }
  if (thread_index == 0) {
    *out = shared_data[0];
  }
}
} // namespace sxt::algr
