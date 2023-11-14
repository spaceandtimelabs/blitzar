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
#include "sxt/scalar25/operation/inner_product.h"

#include <algorithm>

#include "sxt/algorithm/reduction/reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/scalar25/operation/accumulator.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/product_mapper.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// async_inner_product_partial
//--------------------------------------------------------------------------------------------------
static xena::future<s25t::element>
async_inner_product_partial(basct::cspan<s25t::element> lhs,
                            basct::cspan<s25t::element> rhs) noexcept {
  auto n = lhs.size();
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<s25t::element> device_data{&resource};
  auto lhs_fut = xendv::make_active_device_viewable(device_data, lhs);
  auto rhs_dev = co_await xendv::make_active_device_viewable(device_data, rhs);
  auto lhs_dev = co_await std::move(lhs_fut);
  co_return co_await algr::reduce<accumulator>(std::move(stream),
                                               product_mapper{lhs_dev.data(), rhs_dev.data()},
                                               static_cast<unsigned int>(n));
}

//--------------------------------------------------------------------------------------------------
// inner_product
//--------------------------------------------------------------------------------------------------
void inner_product(s25t::element& res, basct::cspan<s25t::element> lhs,
                   basct::cspan<s25t::element> rhs) noexcept {
  auto n = std::min(lhs.size(), rhs.size());
  SXT_DEBUG_ASSERT(n > 0);
  s25o::mul(res, lhs[0], rhs[0]);
  for (size_t i = 1; i < n; ++i) {
    s25o::muladd(res, lhs[i], rhs[i], res);
  }
}

//--------------------------------------------------------------------------------------------------
// async_inner_product_impl
//--------------------------------------------------------------------------------------------------
xena::future<s25t::element> async_inner_product_impl(basct::cspan<s25t::element> lhs,
                                                     basct::cspan<s25t::element> rhs,
                                                     size_t split_factor) noexcept {
  auto n = std::min(lhs.size(), rhs.size());
  SXT_DEBUG_ASSERT(n > 0);
  s25t::element res = s25t::element::identity();
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}, split_factor);
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        auto partial_res = co_await async_inner_product_partial(lhs.subspan(rng.a(), rng.size()),
                                                                rhs.subspan(rng.a(), rng.size()));
        s25o::add(res, res, partial_res);
      });
  co_return res;
}

//--------------------------------------------------------------------------------------------------
// async_inner_product
//--------------------------------------------------------------------------------------------------
xena::future<s25t::element> async_inner_product(basct::cspan<s25t::element> lhs,
                                                basct::cspan<s25t::element> rhs) noexcept {
  auto n = std::min(lhs.size(), rhs.size());
  SXT_DEBUG_ASSERT(n > 0);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<s25t::element> device_data{&resource};
  size_t buffer_size = 0;
  auto is_device_lhs = basdv::is_active_device_pointer(lhs.data());
  auto is_device_rhs = basdv::is_active_device_pointer(rhs.data());
  buffer_size = (static_cast<size_t>(!is_device_lhs) + static_cast<size_t>(!is_device_rhs)) * n;
  if (buffer_size > 0) {
    device_data.resize(buffer_size);
  }
  auto data = device_data.data();
  if (!is_device_lhs) {
    basdv::async_copy_host_to_device(basct::span<s25t::element>{data, n}, lhs.subspan(0, n),
                                     stream);
    lhs = {data, n};
    data += n;
  }
  if (!is_device_rhs) {
    basdv::async_copy_host_to_device(basct::span<s25t::element>{data, n}, rhs.subspan(0, n),
                                     stream);
    rhs = {data, n};
  }
  return algr::reduce<accumulator>(std::move(stream), product_mapper{lhs.data(), rhs.data()},
                                   static_cast<unsigned int>(n));
}

xena::future<s25t::element> async_inner_product2(basct::cspan<s25t::element> lhs,
                                                 basct::cspan<s25t::element> rhs) noexcept {
  (void)async_inner_product_partial;
  (void)lhs;
  (void)rhs;
  return {};
}
} // namespace sxt::s25o
