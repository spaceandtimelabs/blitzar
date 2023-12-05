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

#include <tuple>

#include "sxt/algorithm/base/transform_functor.h"
#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/chunk_options.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/value_type.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_copy.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/chained_resource.h"

namespace sxt::algi {
//--------------------------------------------------------------------------------------------------
// apply_transform_functor
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class Ptrs, class F, size_t... Indexes>
CUDA_CALLABLE void apply_transform_functor(const Ptrs& ptrs, const F& f, unsigned i,
                                           std::index_sequence<Indexes...>) noexcept {
  f(std::get<Indexes>(ptrs)[i]...);
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// transform_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class F, class SpansSrc, size_t... Indexes, class... Args>
xena::future<> transform_impl(basct::span<T> res, F make_f, const SpansSrc& srcs,
                              const basit::index_range& rng, std::index_sequence<Indexes...>,
                              const Args&... xs) noexcept {
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memr::chained_resource alloc{&resource};

  auto dst_futs = std::make_tuple(
      xendv::winked_device_copy(&alloc, std::get<Indexes>(srcs).subspan(rng.a(), rng.size()))...);
  auto f = co_await make_f(&alloc, stream);
  auto dsts = std::make_tuple(co_await std::move(std::get<Indexes>(dst_futs))...);

  auto ptrs = std::make_tuple(std::get<Indexes>(dsts).data()...);
  auto fp = [ptrs, f] __device__ __host__(unsigned /*n*/, unsigned i) noexcept {
    apply_transform_functor(ptrs, f, i, std::make_index_sequence<sizeof...(Indexes)>{});
  };
  launch_for_each_kernel(stream, fp, static_cast<unsigned>(res.size()));
  basdv::async_copy_device_to_host(res, std::get<0>(dsts), stream);
  co_await xendv::await_and_own_stream(std::move(stream));
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// transform
//--------------------------------------------------------------------------------------------------
/**
 * Use multiple GPUs to apply a transformation of contiguous regions of memory
 */
template <class F, class Arg1, class... ArgsRest>
  requires algb::transform_functor_factory<F, bast::value_type_t<Arg1>,
                                           bast::value_type_t<ArgsRest>...>
xena::future<> transform(basct::span<bast::value_type_t<Arg1>> res,
                         basit::chunk_options chunk_options, F make_f, const Arg1& x1,
                         const ArgsRest&... xrest) noexcept {
  auto n = res.size();
  SXT_DEBUG_ASSERT(x1.size() == n && ((xrest.size() == n) && ...));
  if (n == 0) {
    co_return;
  }
  std::tuple<basct::cspan<bast::value_type_t<Arg1>>, basct::cspan<bast::value_type_t<ArgsRest>>...>
      srcs{x1, xrest...};
  auto full_rng = basit::index_range{0, n}
                 .min_chunk_size(chunk_options.min_size)
                 .max_chunk_size(chunk_options.max_size);
  co_await xendv::concurrent_for_each(
      full_rng, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        co_await detail::transform_impl(res.subspan(rng.a(), rng.size()), make_f, srcs, rng,
                                        std::make_index_sequence<sizeof...(ArgsRest) + 1>{});
      });
}

template <class F, class Arg1, class... ArgsRest>
  requires algb::transform_functor<F, bast::value_type_t<Arg1>, bast::value_type_t<ArgsRest>...>
xena::future<> transform(basct::span<bast::value_type_t<Arg1>> res,
                         basit::chunk_options chunk_options, F f, const Arg1& x1,
                         const ArgsRest&... xrest) noexcept {
  auto make_f = [&](std::pmr::polymorphic_allocator<> /*alloc*/, basdv::stream& /*stream*/) {
    return xena::make_ready_future<F>(F{f});
  };
  co_await transform(res, chunk_options, make_f, x1, xrest...);
}
} // namespace sxt::algi
