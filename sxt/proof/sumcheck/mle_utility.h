/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/field/element.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/device/to_device_copier.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// copy_partial_mles
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
void copy_partial_mles(memmg::managed_array<T>& partial_mles, basdv::stream& stream,
                       basct::cspan<T> mles, unsigned n, unsigned a, unsigned b) noexcept {
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto mid = 1u << (num_variables - 1u);
  auto num_mles = mles.size() / n;
  auto part1_size = b - a;
  SXT_DEBUG_ASSERT(a < b && b <= n);
  auto ap = std::min(mid + a, n);
  auto bp = std::min(mid + b, n);
  auto part2_size = bp - ap;

  // resize array
  auto partial_length = part1_size + part2_size;
  partial_mles.resize(partial_length * num_mles);

  // copy data
  for (unsigned mle_index = 0; mle_index < num_mles; ++mle_index) {
    // first part
    auto src = mles.subspan(n * mle_index + a, part1_size);
    auto dst = basct::subspan(partial_mles, partial_length * mle_index, part1_size);
    basdv::async_copy_host_to_device(dst, src, stream);

    // second part
    src = mles.subspan(n * mle_index + ap, part2_size);
    dst = basct::subspan(partial_mles, partial_length * mle_index + part1_size, part2_size);
    if (!src.empty()) {
      basdv::async_copy_host_to_device(dst, src, stream);
    }
  }
}

template <basfld::element T>
xena::future<> copy_partial_mles2(memmg::managed_array<T>& partial_mles, basdv::stream& stream,
                                  basct::cspan<T> mles, unsigned n, unsigned a,
                                  unsigned b) noexcept {
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto mid = 1u << (num_variables - 1u);
  auto num_mles = mles.size() / n;
  auto part1_size = b - a;
  SXT_DEBUG_ASSERT(a < b && b <= n);
  auto ap = std::min(mid + a, n);
  auto bp = std::min(mid + b, n);
  auto part2_size = bp - ap;

  // copier
  auto partial_length = part1_size + part2_size;
  partial_mles.resize(partial_length * num_mles);
  xendv::to_device_copier copier{partial_mles, stream};

  // copy data
  for (unsigned mle_index = 0; mle_index < num_mles; ++mle_index) {
    // first part
    auto src = mles.subspan(n * mle_index + a, part1_size);
    co_await xendv::copy(copier, src);

    // second part
    src = mles.subspan(n * mle_index + ap, part2_size);
    co_await xendv::copy(copier, src);
  }
}

//--------------------------------------------------------------------------------------------------
// copy_folded_mles
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
void copy_folded_mles(basct::span<T> host_mles, basdv::stream& stream, basct::cspan<T> device_mles,
                      unsigned np, unsigned a, unsigned b) noexcept {
  auto num_mles = host_mles.size() / np;
  auto slice_n = device_mles.size() / num_mles;
  auto slice_np = b - a;
  SXT_DEBUG_ASSERT(
      // clang-format off
      host_mles.size() == num_mles * np && 
      device_mles.size() == num_mles * slice_n &&
      b <= np
      // clang-format on
  );
  for (unsigned mle_index = 0; mle_index < num_mles; ++mle_index) {
    auto src = device_mles.subspan(mle_index * slice_n, slice_np);
    auto dst = host_mles.subspan(mle_index * np + a, slice_np);
    basdv::async_copy_device_to_host(dst, src, stream);
  }
}

//--------------------------------------------------------------------------------------------------
// get_gpu_memory_fraction
//--------------------------------------------------------------------------------------------------
template <basfld::element T> double get_gpu_memory_fraction(basct::cspan<T> mles) noexcept {
  auto total_memory = static_cast<double>(basdv::get_total_device_memory());
  return static_cast<double>(mles.size() * sizeof(T)) / total_memory;
}
} // namespace sxt::prfsk
