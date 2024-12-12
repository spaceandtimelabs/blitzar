/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include <algorithm>
#include <iostream>
#include <limits>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/log/log.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/bucket_method2/constants.h"
#include "sxt/multiexp/bucket_method2/reduce.h"
#include "sxt/multiexp/bucket_method2/sum.h"

namespace sxt::mtxbk2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> generators,
                                 basct::cspan<const uint8_t*> exponents,
                                 unsigned element_num_bytes) noexcept {
  if (res.empty()) {
    co_return;
  }
  basl::info("compute a multiexponentiation with {} outputs of length {}", res.size(),
             generators.size());
  auto num_outputs = static_cast<unsigned>(res.size());
  static constexpr unsigned bit_width = 8;
  static constexpr unsigned num_buckets_per_digit = (1u << bit_width) - 1u;
  static constexpr unsigned num_digits = 32;
  static constexpr unsigned num_buckets_per_output = num_buckets_per_digit * num_digits;
  const unsigned num_buckets_total = num_buckets_per_output * num_outputs;

  // accumulate
  memmg::managed_array<T> sums{num_buckets_total, memr::get_device_resource()};
  co_await sum_buckets<T>(sums, generators, exponents, element_num_bytes, bit_width);

  // reduce bucket sums
  basl::info("reducing {} buckets into {} outputs", sums.size(), num_outputs);
  co_await reduce_buckets<T>(res, sums, element_num_bytes, bit_width);
  basl::info("finished multiexponentiation with {} outputs of length {}", res.size(),
             generators.size());
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate
//--------------------------------------------------------------------------------------------------
/**
 * Attempt to compute a multi-exponentiation using the bucket method if the problem dimensions
 * suggest it will give a performance benefit; otherwise, return an empty array.
 *
 * This version of the bucket method targets cases where the multiexponention length
 * is shorter and the number of outputs is larger.
 */
template <bascrv::element T>
xena::future<memmg::managed_array<T>>
try_multiexponentiate(basct::cspan<T> generators,
                      basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_outputs = exponents.size();
  memmg::managed_array<T> res{memr::get_pinned_resource()};
  uint64_t min_n = std::numeric_limits<uint64_t>::max();
  uint64_t max_n = 0;
  for (auto& exponent : exponents) {
    if (exponent.element_nbytes != 32) {
      co_return res;
    }
    min_n = std::min(min_n, exponent.n);
    max_n = std::max(max_n, exponent.n);
  }
  if (min_n != max_n) {
    co_return res;
  }
  auto n = max_n;
  if (n > max_multiexponentiation_length_v || n < 256) {
    co_return res;
  }
  SXT_DEBUG_ASSERT(generators.size() >= n);
  generators = generators.subspan(0, n);
  res.resize(num_outputs);
  memmg::managed_array<const uint8_t*> exponents_p(num_outputs);
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    exponents_p[output_index] = exponents[output_index].data;
  }
  co_await xendv::concurrent_for_each(
      basit::index_range{0, num_outputs}, [&](const basit::index_range& rng) -> xena::future<> {
        auto exponents_slice = basct::subspan(exponents_p, rng.a(), rng.size());
        auto res_slice = basct::subspan(res, rng.a(), rng.size());
        co_await multiexponentiate<T>(res_slice, generators, exponents_slice, 32);
      });
  co_return res;
}
} // namespace sxt::mtxbk2
