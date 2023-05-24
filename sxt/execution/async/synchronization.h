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

#include <concepts>
#include <memory>
#include <type_traits>

#include "sxt/base/device/event.h"
#include "sxt/base/device/event_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/gpu_computation_event.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/schedule/scheduler.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// await_stream
//--------------------------------------------------------------------------------------------------
future<> await_stream(bast::raw_stream_t stream) noexcept;

template <class T>
  requires std::constructible_from<std::optional<std::remove_cvref_t<T>>, T&&>
future<std::remove_cvref_t<T>> await_stream(T&& val, bast::raw_stream_t stream) noexcept {
  using Tp = std::remove_cvref_t<T>;
  future_state<Tp> state;
  state.emplace(std::forward<T>(val));
  promise<Tp> p;
  future<Tp> res{p, std::move(state)};
  basdv::event event;
  basdv::record_event(event, stream);
  xens::get_scheduler().schedule(
      std::make_unique<gpu_computation_event<Tp>>(std::move(event), std::move(p)));
  return res;
}

//--------------------------------------------------------------------------------------------------
// await_and_own_stream
//--------------------------------------------------------------------------------------------------
future<> await_and_own_stream(basdv::stream&& stream) noexcept;

template <class T>
  requires std::constructible_from<std::optional<std::remove_cvref_t<T>>, T&&>
future<std::remove_cvref_t<T>> await_and_own_stream(basdv::stream&& stream, T&& val) noexcept {
  using Tp = std::remove_cvref_t<T>;
  future_state<Tp> state;
  state.emplace(std::forward<T>(val));
  promise<Tp> p;
  future<Tp> res{p, std::move(state)};
  basdv::event event;
  basdv::record_event(event, stream);
  computation_handle handle;
  handle.add_stream(std::move(stream));
  xens::get_scheduler().schedule(std::make_unique<gpu_computation_event<Tp>>(
      std::move(event), std::move(handle), std::move(p)));
  return res;
}
} // namespace sxt::xena
