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
#include "sxt/execution/device/for_each.h"

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/state.h"
#include "sxt/base/iterator/split.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/available_device.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// for_each_device_impl
//--------------------------------------------------------------------------------------------------
static xena::future<> for_each_device_impl(
    chunk_context* ctx, chunk_context* ctx_p, unsigned& chunk_index,
    basit::index_range_iterator& iter, basit::index_range_iterator last,
    std::function<xena::future<>(const chunk_context& ctx, basit::index_range)> f) noexcept {
  auto device_index = ctx->device_index;
  while (true) {
    if (iter == last) {
      co_await ctx_p->alt_future;
      co_return;
    }
    auto chunk = *iter++;
    ctx_p->chunk_index = chunk_index++;
    basdv::set_device(device_index);
    auto fut = f(*ctx_p, chunk);
    co_await ctx_p->alt_future;
    ctx->alt_future = std::move(fut);
    std::swap(ctx, ctx_p);
  }
  co_await ctx->alt_future;
}

//--------------------------------------------------------------------------------------------------
// concurrent_for_each
//--------------------------------------------------------------------------------------------------
xena::future<>
concurrent_for_each(basit::index_range_iterator first, basit::index_range_iterator last,
                    std::function<xena::future<>(const basit::index_range&)> f) noexcept {
  std::vector<xena::future<>> futs;
  futs.reserve(std::distance(first, last));
  for (auto it = first; it != last; ++it) {
    auto device = co_await await_available_device();
    basdv::active_device_guard guard{device};
    futs.emplace_back(f(*it));
  }
  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}

xena::future<>
concurrent_for_each(basit::index_range rng,
                    std::function<xena::future<>(const basit::index_range&)> f) noexcept {
  basit::split_options split_options{
      .split_factor = basdv::get_num_devices(),
  };
  auto [first, last] = basit::split(rng, split_options);
  return concurrent_for_each(first, last, f);
}

//--------------------------------------------------------------------------------------------------
// for_each_device
//--------------------------------------------------------------------------------------------------
xena::future<> for_each_device(
    basit::index_range_iterator first, basit::index_range_iterator last,
    std::function<xena::future<>(const chunk_context& ctx, basit::index_range)> f) noexcept {
  if (first == last) {
    co_return;
  }

  unsigned chunk_index = 0;
  auto num_chunks = static_cast<unsigned>(std::distance(first, last));
  auto num_devices = basdv::get_num_devices();
  auto num_devices_used = static_cast<unsigned>(std::min(num_chunks, num_devices));

  basdv::active_device_guard guard;

  // set up contexts
  std::vector<chunk_context> contexts(num_devices_used);
  for (unsigned device_index = 0; device_index < num_devices_used; ++device_index) {
    auto& ctx = contexts[device_index];
    ctx.device_index = device_index;
    ctx.alt_future = xena::make_ready_future();
    ctx.num_devices_used = num_devices_used;
  }
  std::vector<chunk_context> contexts_p(contexts);

  // initial launches
  for (unsigned device_index = 0; device_index < num_devices_used; ++device_index) {
    auto& ctx = contexts[device_index];
    ctx.chunk_index = chunk_index++;
    auto chunk = *first++;
    basdv::set_device(device_index);
    contexts_p[device_index].alt_future = f(ctx, chunk);
  }

  // continue launching until all chunks are processed
  std::vector<xena::future<>> futs(num_devices_used);
  for (unsigned device_index = 0; device_index < num_devices_used; ++device_index) {
    futs[device_index] = for_each_device_impl(&contexts[device_index], &contexts_p[device_index],
                                              chunk_index, first, last, f);
  }

  // wait for everything to finish
  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}
} // namespace sxt::xendv
