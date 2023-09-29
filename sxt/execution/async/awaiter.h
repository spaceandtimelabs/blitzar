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

#include <optional>
#include <type_traits>

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/property.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine_promise.h"
#include "sxt/execution/async/future.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// awaiter
//--------------------------------------------------------------------------------------------------
template <class T> class awaiter {
public:
  explicit awaiter(future<T>&& fut) noexcept : future_{std::move(fut)} {
    if (basdv::get_num_devices() > 0) {
      active_guard_.emplace();
    }
  }

  bool await_ready() const noexcept { return future_.ready(); }

  template <class U>
  void await_suspend(std::coroutine_handle<coroutine_promise<U>> handle) noexcept {
    SXT_DEBUG_ASSERT(!future_.ready(), "we don't support preemptive futures");
    auto pr = future_.promise();
    SXT_DEBUG_ASSERT(pr != nullptr, "future must have an attached promise");
    pr->set_continuation(handle.promise());
  }

  T await_resume() noexcept {
    if constexpr (std::is_same_v<T, void>) {
      return;
    } else {
      return T{std::move(future_.value())};
    }
  }

private:
  future<T> future_;
  std::optional<basdv::active_device_guard> active_guard_;
};

// Disable explicit instantiation. Workaround to
// https://developer.nvidia.com/bugs/4288496
/* extern template class awaiter<void>; */
} // namespace sxt::xena
