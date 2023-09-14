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

#include <memory>
#include <optional>
#include <utility>

#include "sxt/base/device/event.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/computation_event.h"
#include "sxt/execution/device/computation_handle.h"
#include "sxt/execution/schedule/scheduler.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// event_future
//--------------------------------------------------------------------------------------------------
template <class T> class event_future {
public:
  using value_type = T;

  event_future() noexcept = default;

  ~event_future() noexcept {
    if (event_) {
      (void)xena::future<T>{std::move(*this)};
    }
  }

  explicit event_future(T&& value) noexcept : value_{std::move(value)} {}

  event_future(T&& value, int device, basdv::event&& event, computation_handle handle) noexcept
      : value_{std::move(value)}, device_{device}, event_{std::move(event)},
        handle_{std::move(handle)} {}

  const T& value() const noexcept { return value_; }
  T& value() noexcept { return value_; }

  const std::optional<basdv::event>& event() const noexcept { return event_; }

  operator xena::future<T>() && noexcept {
    if (!event_) {
      return xena::make_ready_future<T>(std::move(value_));
    }
    SXT_DEBUG_ASSERT(device_ >= 0);
    xena::future_state<T> state;
    state.emplace(std::move(value_));
    xena::promise<T> p;
    xena::future<T> res{p, std::move(state)};
    xens::get_scheduler().schedule(std::make_unique<computation_event<T>>(
        device_, std::move(*event_), std::move(handle_), std::move(p)));
    event_.reset();
    return res;
  }

private:
  T value_;
  int device_{-1};
  std::optional<basdv::event> event_;
  computation_handle handle_;
};
} // namespace sxt::xendv
