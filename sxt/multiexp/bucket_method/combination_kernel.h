#pragma once

#include "sxt/base/num/divide_up.h"
#include "sxt/algorithm/base/reducer.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_accumulate
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer>
__global__ void bucket_accumulate(typename Reducer::value_type* bucket_sums,
                                  const unsigned* lengths) {
  auto bucket_index = blockIdx.x;
  auto output_index = blockIdx.y;
  (void)bucket_index;
  (void)output_index;
  (void)bucket_sums;
  (void)lengths;
}
} // namespace sxt::mtxbk
