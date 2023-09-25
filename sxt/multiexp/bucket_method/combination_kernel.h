#pragma once

#include "sxt/algorithm/base/reducer.h"
#include "sxt/algorithm/reduction/warp_reduction.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// combine_partial_bucket_sums
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer>
__global__ void combine_partial_bucket_sums(typename Reducer::value_type* out,
                                            typename Reducer::value_type* partial_bucket_sums,
                                            unsigned num_partial_buckets) {
  using T = typename Reducer::value_type;
  auto bucket_group_size = gridDim.x;
  auto bucket_group_index = threadIdx.x;
  auto num_bucket_groups = blockDim.x;
  auto bucket_index = blockIdx.x;
  auto output_index = blockIdx.y;

  partial_bucket_sums += bucket_index + bucket_group_size * bucket_group_index +
                         bucket_group_size * num_bucket_groups * num_partial_buckets * output_index;

  out += bucket_index + bucket_group_size * bucket_group_index +
         bucket_group_size * num_bucket_groups * output_index;

  T sum = *partial_bucket_sums;
  partial_bucket_sums += bucket_group_size * num_bucket_groups;
  for (unsigned i = 1; i < num_partial_buckets; ++i) {
    Reducer::accumulate_inplace(sum, *partial_bucket_sums);
    partial_bucket_sums += bucket_group_size * num_bucket_groups;
  }

  *out = sum;
}
} // namespace sxt::mtxbk
