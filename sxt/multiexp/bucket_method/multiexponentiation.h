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
#include "sxt/multiexp/bucket_method/accumulation.h"
#include "sxt/multiexp/bucket_method/combination.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> generators,
                                 basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr unsigned bucket_group_size = 255;
  constexpr unsigned num_bucket_groups = 32;
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(res.size() == num_outputs);
  if (res.empty()) {
    co_return;
  }

  basdv::stream stream;

  // accumulate
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> bucket_sums{bucket_group_size * num_bucket_groups * num_outputs,
                                      &resource};
  co_await accumulate_buckets<T>(bucket_sums, generators, exponents);

  // reduce buckets
  memmg::managed_array<T> reduced_buckets_dev{bucket_group_size * num_outputs, &resource};
  static unsigned num_threads = 32;
  dim3 block_dims(basn::divide_up(bucket_group_size, num_threads), num_outputs, 1);
  combine_bucket_groups<bucket_group_size, num_bucket_groups>
      <<<block_dims, num_threads, 0, stream>>>(reduced_buckets_dev.data(), bucket_sums.data());
  memmg::managed_array<T> reduced_buckets{reduced_buckets_dev.size(), memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(reduced_buckets, reduced_buckets_dev, stream);
  co_await xendv::await_stream(stream);
  reduced_buckets_dev.reset();

  // combine buckets
  combine_buckets<T>(res, reduced_buckets);
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
try_multiexponentiate(basct::cspan<Element> generators,
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
  static constexpr size_t max_output_chunk = 8;
  auto output_chunks =
      basit::split(basit::index_range{0, num_outputs}.max_chunk_size(max_output_chunk), 1);
  for (auto chunk_iter = output_chunks.first; chunk_iter != output_chunks.second; ++chunk_iter) {
    auto chunk = *chunk_iter;
    co_await multiexponentiate<Element>(basct::subspan(res, chunk.a(), chunk.size()), generators,
                                        basct::subspan(exponents_p, chunk.a(), chunk.size()));
  }
  co_return res;
}
} // namespace sxt::mtxbk
