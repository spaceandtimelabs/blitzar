#pragma once

#include <algorithm>
#include <limits>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate2
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate2(basct::span<T> res, basct::cspan<T> generators,
                                 basct::cspan<const uint8_t*> exponents) noexcept {
  auto num_outputs = static_cast<unsigned>(res.size());
  static constexpr unsigned bit_width = 8;
  static constexpr unsigned num_buckets_per_digit = (1u << bit_width) - 1u;
  static constexpr unsigned num_digits = 32;
  static constexpr unsigned num_buckets_per_output = num_buckets_per_digit * num_digits;
  static const unsigned num_buckets_total = num_buckets_per_output * num_outputs;

  // accumulate
  memmg::managed_array<T> bucket_sums{num_buckets_total, memr::get_pinned_resource()};
  (void)bucket_sums;

  // reduce bucket sums
  (void)res;
  (void)generators;
  (void)exponents;
  return xena::make_ready_future();
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate2
//--------------------------------------------------------------------------------------------------
/**
 * Attempt to compute a multi-exponentiation using the bucket method if the problem dimensions
 * suggest it will give a performance benefit; otherwise, return an empty array.
 */
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
try_multiexponentiate2(basct::cspan<Element> generators,
                       basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_outputs = exponents.size();
  memmg::managed_array<Element> res;
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
  SXT_DEBUG_ASSERT(generators.size() >= n);
  generators = generators.subspan(0, n);
  memmg::managed_array<const uint8_t*> exponents_p(num_outputs);
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    exponents_p[output_index] = exponents[output_index].data;
  }
  res.resize(num_outputs);
  co_await multiexponentiate2(res, generators, exponents_p);
  co_return res;
}
} // namespace sxt::mtxbk
