#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// combine_buckets_impl 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine_buckets_impl(T& sum, basct::cspan<T> bucket_sums) noexcept {
  (void)sum;
  (void)bucket_sums;
}

//--------------------------------------------------------------------------------------------------
// combine_buckets 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine_buckets(basct::span<T> sums, basct::cspan<T> bucket_sums) noexcept {
  auto num_outputs = sums.size();
  auto bucket_group_size = bucket_sums.size() / num_outputs;
  SXT_ASSERT(
      // clang-format off
      sums.size() == num_outputs && 
      bucket_sums.size() == num_outputs * bucket_group_size
      // clang-format on
  );
  for (size_t output_index=0; output_index<num_outputs; ++output_index) {
    combine_buckets_impl(sums[output_index],
                         bucket_sums.subspan(bucket_group_size * output_index, bucket_group_size));
  }
}
} // namespace sxt::mtxbk
