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
#include <iterator>
#include <optional>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/event.h"
#include "sxt/base/device/event_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/bucket_method/multiexponentiation.h"
#include "sxt/multiexp/curve/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/curve/multiproduct.h"
#include "sxt/multiexp/curve/multiproducts_combination.h"
#include "sxt/multiexp/curve/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/pippenger/multiproduct_decomposition_gpu.h"

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_utility.h"
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
#include "sxt/multiexp/bucket_method2/multiproduct_table_kernel.h"
#include "sxt/multiexp/bucket_method2/reduce.h"
#include "sxt/multiexp/bucket_method2/sum.h"

namespace sxt::mtxcrv {
//--------------------------------------------------------------------------------------------------
// async_compute_multiexponentiation_impl
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
static xena::future<> async_compute_multiexponentiation_impl(
    memmg::managed_array<Element>& products, basct::span<uint8_t> or_all,
    const xendv::event_future<basct::cspan<Element>>& generators_event,
    mtxb::exponent_sequence exponents) noexcept {
  auto num_bytes = exponents.element_nbytes;
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  bool is_signed = static_cast<bool>(exponents.is_signed);

  // decompose exponents
  memmg::managed_array<unsigned> indexes{&resource};
  memmg::managed_array<unsigned> product_sizes(num_bytes * 8u);
  co_await mtxpi::compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
  if (indexes.empty()) {
    co_return;
  }

  // or_all_p
  std::vector<uint8_t> or_all_p(num_bytes);
  size_t bit_index = 0;
  for (size_t byte_index = 0; byte_index < num_bytes; ++byte_index) {
    uint8_t val = 0;
    for (int i = 0; i < 8; ++i) {
      val |= static_cast<uint8_t>(product_sizes[bit_index++] > 0) << i;
    }
    or_all_p[byte_index] = val;
  }

  // compute multiproduct
  auto last = std::remove(product_sizes.begin(), product_sizes.end(), 0u);
  product_sizes.shrink(static_cast<size_t>(std::distance(product_sizes.begin(), last)));
  memmg::managed_array<Element> products_p(product_sizes.size());
  xendv::synchronize_event(stream, generators_event);
  co_await async_compute_multiproduct<Element>(products_p, stream, generators_event.value(),
                                               indexes, product_sizes, is_signed);
  fold_multiproducts<Element>(products, or_all, products_p, or_all_p);
}

//--------------------------------------------------------------------------------------------------
// async_compute_multiexponentiation_partial
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
static xena::future<> async_compute_multiexponentiation_partial(
    basct::span<basct::blob_array> or_alls, basct::span<memmg::managed_array<Element>> products,
    basct::cspan<Element> generators, basct::cspan<mtxb::exponent_sequence> exponents,
    basit::index_range rng) noexcept {
  auto num_outputs = or_alls.size();
  // set up generators
  memmg::managed_array<Element> generators_data{memr::get_device_resource()};
  auto generators_event = xendv::make_active_device_viewable(
      generators_data,
      generators.subspan(static_cast<size_t>(rng.a()), static_cast<size_t>(rng.size())));
  std::vector<xena::future<>> futs;
  futs.reserve(num_outputs);
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    auto output_exponents = exponents[output_index];
    if (static_cast<size_t>(rng.a()) >= output_exponents.n) {
      continue;
    }
    output_exponents.data += output_exponents.element_nbytes * rng.a();
    output_exponents.n = std::min(static_cast<size_t>(rng.size()), output_exponents.n - rng.a());
    auto fut = async_compute_multiexponentiation_impl(
        products[output_index], or_alls[output_index][0], generators_event, output_exponents);
    futs.emplace_back(std::move(fut));
  }

  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
memmg::managed_array<Element>
compute_multiexponentiation(basct::cspan<Element> generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  pippenger_multiproduct_solver<Element> solver;
  multiexponentiation_cpu_driver<Element> driver{&solver};
  // Note: the cpu driver is non-blocking so that the future upon return the future is
  // available
  return mtxpi::compute_multiexponentiation(
             driver,
             {static_cast<const void*>(generators.data()), generators.size(), sizeof(Element)},
             exponents)
      .value()
      .template as_array<Element>();
}

//--------------------------------------------------------------------------------------------------
// async_compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
async_compute_multiexponentiation(basct::cspan<Element> generators,
                                  basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  // try bucket method first
  /* auto res_maybe = co_await mtxbk2::try_multiexponentiate(generators, exponents); */
  auto res_maybe = co_await mtxbk::try_multiexponentiate(generators, exponents);
  /* if (res_maybe.empty()) { */
  /*   res_maybe = co_await mtxbk::try_multiexponentiate(generators, exponents); */
  /* } */
  if (!res_maybe.empty()) {
    co_return res_maybe;
  }

  // use more general method
  auto num_outputs = exponents.size();
  std::vector<basct::blob_array> or_alls;
  or_alls.reserve(num_outputs);
  for (auto& exponent_sequence : exponents) {
    or_alls.emplace_back(1, exponent_sequence.element_nbytes);
  }
  std::vector<memmg::managed_array<Element>> products(num_outputs);

  // Pick some reasonable values for min and max chunk size so that
  // we don't run out of GPU memory or split computations that are
  // too small.
  //
  // Note: These haven't been informed by much benchmarking. I'm
  // sure there are better values. This is just putting in some
  // ballpark estimates to get started.
  size_t min_chunk_size = 1ull << 10u;
  size_t max_chunk_size = 1ull << 20u;
  if (num_outputs > 0) {
    max_chunk_size = basn::divide_up(max_chunk_size, num_outputs);
    min_chunk_size *= num_outputs;
    min_chunk_size = std::min(max_chunk_size, min_chunk_size);
  }
  auto rng = basit::index_range{0, generators.size()}
                 .min_chunk_size(min_chunk_size)
                 .max_chunk_size(max_chunk_size);
  co_await xendv::concurrent_for_each(rng, [&](const basit::index_range& rng) noexcept {
    return async_compute_multiexponentiation_partial<Element>(or_alls, products, generators,
                                                              exponents, rng);
  });
  memmg::managed_array<Element> res(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    combine_multiproducts<Element>({&res[i], 1}, or_alls[i], products[i]);
  }
  co_return res;
}

template <bascrv::element Element>
xena::future<Element>
async_compute_multiexponentiation(basct::cspan<Element> generators,
                                  const mtxb::exponent_sequence& exponents) noexcept {
  auto res = co_await async_compute_multiexponentiation<Element>(generators, {&exponents, 1});
  co_return res[0];
}
} // namespace sxt::mtxcrv
