#pragma once

#include "sxt/base/num/divide_up.h"
#include "sxt/algorithm/base/reducer.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_accumulate
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer>
__global__ void bucket_accumulate(typename Reducer::value_type* bucket_sums,
                                  const unsigned* num_partial_buckets) {
  auto bucket_index = blockIdx.x;
  auto num_buckets_per_generator = gridDim.x;
  auto output_index = blockIdx.y;
  for (unsigned i=0; i<output_index; ++i) {
    bucket_sums += *num_partial_buckets++ * num_buckets_per_generator;
  }
  auto n = *num_partial_buckets;
  bucket_sums += bucket_index;
  (void)bucket_index;
  (void)output_index;
  (void)bucket_sums;
}
} // namespace sxt::mtxbk
