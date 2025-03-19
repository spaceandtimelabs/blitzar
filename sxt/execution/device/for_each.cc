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
/**
 * Invoke the function f on the range of chunks provided, splitting the work across available
 * devices.
 */
xena::future<> for_each_device(
    basit::index_range_iterator first, basit::index_range_iterator last,
    std::function<xena::future<>(device_context& ctx, const basit::index_range&)> f) noexcept {
  basdv::active_device_guard guard;

  auto num_chunks = static_cast<unsigned>(std::distance(first, last));
  auto num_devices = basdv::get_num_devices();
  auto num_devices_used = std::min(num_chunks, num_devices);

  std::vector<device_context> contexts(num_devices_used);

  // initial launch
  for (unsigned device_index=0; device_index<num_devices_used; ++device_index) {
    basdv::set_device(device_index);
    auto& ctx = contexts[device_index];
    ctx.device_index = device_index;
    ctx.num_devices_used = num_devices_used;
    ctx.alt_future = xena::make_ready_future();
    ctx.alt_future2 = xena::shared_future<>{xena::make_ready_future()};
    auto chunk = *first++;
    ctx.alt_future = f(ctx, chunk);
  }

  // alternate launch
  std::vector<xena::future<>> futs;
  futs.reserve(num_devices_used);
  for (unsigned device_index=0; device_index<num_devices_used; ++device_index) {
    if (first == last) {
      break;
    }
    basdv::set_device(device_index);
    auto chunk = *first++;
    auto fut = f(contexts[device_index], chunk);
    futs.emplace_back(std::move(fut));
  }

  // continue launches until all chunks are processed
  while (first != last) {
    for (unsigned device_index = 0; device_index < num_devices_used; ++device_index) {
      if (first == last) {
        break;
      }
      basdv::set_device(device_index);
      auto chunk = *first++;
      auto& ctx = contexts[device_index];
      co_await std::move(ctx.alt_future);
      ctx.alt_future = std::move(futs[device_index]);
      futs[device_index] = f(ctx, chunk);
    }
  }

  // wait for everything to finish
  for (auto& ctx : contexts) {
    co_await std::move(ctx.alt_future);
  }
  for (auto& fut : futs) {
    co_await std::move(fut);
  }

  // start futures for
  //   device_1, ..., device_m
  // fut_1, ..., fut_m
  //
  //   f() {
  //      while (!chunks.empty()) {
  //        chunk <- get_chunk()
  //        stream s;
  //        copy_memory(s, chunk)
  //        invoke kernel(s, chunk)
  //        yield;
  //        chunk_p <- get_chunk()
  //        stream sp;
  //        copy_memory(sp, chunk_p);
  //        await s
  //        invoke_kernel(sp, chunk_p)
  //      }
  //   }
  //
  //  // if given a functor like this, can I do the rest of the management code?
  //  [](stream& s, stream& sp, chunk) {
  //     copy_memory(s, chunk);
  //     await sp;
  //     invoke_kernel(s, chunk);
  //  }
  (void)first;
  (void)last;
  (void)f;
  
}
} // namespace sxt::xendv
