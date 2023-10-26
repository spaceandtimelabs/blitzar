/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// combine_buckets_impl
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine_buckets_impl(T& sum, basct::span<T> bucket_sums) noexcept {
  auto i = bucket_sums.size() - 1u;
  T t = bucket_sums[i];
  sum = bucket_sums[i];
  while (i-- > 0) {
    add_inplace(t, bucket_sums[i]);
    add(sum, sum, t);
  }
}

//--------------------------------------------------------------------------------------------------
// combine_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine_buckets(basct::span<T> sums, basct::span<T> bucket_sums) noexcept {
  auto num_outputs = sums.size();
  auto bucket_group_size = bucket_sums.size() / num_outputs;
  SXT_DEBUG_ASSERT(
      // clang-format off
      sums.size() == num_outputs && 
      bucket_sums.size() == num_outputs * bucket_group_size
      // clang-format on
  );
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    combine_buckets_impl(sums[output_index],
                         bucket_sums.subspan(bucket_group_size * output_index, bucket_group_size));
  }
}
} // namespace sxt::mtxbk
