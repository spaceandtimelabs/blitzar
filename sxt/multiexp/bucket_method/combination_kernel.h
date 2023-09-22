#pragma once

#include "sxt/algorithm/base/reducer.h"
#include "sxt/algorithm/reduction/warp_reduction.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// combine_partial_bucket_sums
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer, unsigned BlockSize>
__global__ void combine_partial_bucket_sums(typename Reducer::value_type* out,
                                            typename Reducer::value_type* partial_bucket_sums,
                                            unsigned num_partial_buckets) {
  static_assert(BlockSize <= 32);
  using T = typename Reducer::value_type;
  auto thread_index = threadIdx.x;
  auto bucket_index = blockIdx.x;
  auto num_buckets_per_generator = gridDim.x;
  auto output_index = blockIdx.y;
  partial_bucket_sums += num_partial_buckets * num_buckets_per_generator * output_index;
  partial_bucket_sums += num_partial_buckets * bucket_index;
  __shared__ T shared_data[BlockSize];
  shared_data[thread_index] = partial_bucket_sums[thread_index];
  auto index = thread_index + BlockSize;
  while (index < num_partial_buckets) {
    Reducer::accumulate_inplace(shared_data[thread_index], partial_bucket_sums[index]);
    index += BlockSize;
  }
  algr::warp_reduce<Reducer, BlockSize>(shared_data, thread_index);
  out += num_buckets_per_generator * output_index;
  if (thread_index == 0) {
    out[bucket_index] = shared_data[0];
  }
}
} // namespace sxt::mtxbk
