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

#include <utility>

#include "sxt/base/container/span.h"
#include "sxt/base/device/property.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::basit {
struct split_options;
}

namespace sxt::prfsk {
class device_cache;

//--------------------------------------------------------------------------------------------------
// sum_options
//--------------------------------------------------------------------------------------------------
struct sum_options {
  unsigned min_chunk_size = 100'000u;
  unsigned max_chunk_size = 250'000u;
  unsigned split_factor = unsigned(basdv::get_num_devices());
};

//--------------------------------------------------------------------------------------------------
// sum_gpu
//--------------------------------------------------------------------------------------------------
xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       const basit::split_options& options, basct::cspan<s25t::element> mles,
                       unsigned n) noexcept;

xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       basct::cspan<s25t::element> mles, unsigned n) noexcept;

xena::future<> sum_gpu(basct::span<s25t::element> p, basct::cspan<s25t::element> mles,
                       basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                       basct::cspan<unsigned> product_terms, unsigned n) noexcept;
} // namespace sxt::prfsk
