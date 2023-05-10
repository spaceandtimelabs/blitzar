/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include <utility>

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future_state.h"
#include "sxt/execution/async/promise_future_base.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// promise
//--------------------------------------------------------------------------------------------------
template <class T = void> class promise final : public promise_base {
public:
  promise() noexcept = default;

  promise(const promise&) = delete;
  promise(promise&& other) noexcept
      : promise_base{std::move(static_cast<promise_base&>(other))},
        state_{std::exchange(other.state_, nullptr)},
        continuation_{std::exchange(other.continuation_, nullptr)} {}

  promise& operator=(const promise&) = delete;
  promise& operator=(promise&& other) noexcept {
    SXT_DEBUG_ASSERT(state_ == nullptr);
    SXT_DEBUG_ASSERT(continuation_ == nullptr);
    *static_cast<promise_base*>(this) = std::move(static_cast<promise_base&>(other));
    state_ = std::exchange(other.state_, nullptr);
    continuation_ = std::exchange(other.continuation_, nullptr);
    return *this;
  }

  void set_state(future_state<T>& state) noexcept { state_ = &state; }

  future_state<T>& state() noexcept {
    SXT_DEBUG_ASSERT(state_ != nullptr);
    return *state_;
  }

  void set_continuation(task& cont) noexcept {
    SXT_DEBUG_ASSERT(continuation_ == nullptr);
    continuation_ = &cont;
  }

  template <class... Args>
    requires std::constructible_from<T, Args&&...>
  void set_value(Args&&... args) noexcept {
    this->state().emplace(std::forward<Args>(args)...);
    this->make_ready();
  }

  void make_ready() noexcept {
    SXT_DEBUG_ASSERT(state_ != nullptr && !state_->ready());
    state_->make_ready();
    if (auto fut = this->future(); fut != nullptr) {
      fut->set_promise(nullptr);
    }
    if (continuation_ != nullptr) {
      continuation_->run_and_dispose();
    }
  }

private:
  future_state<T>* state_{nullptr};
  task* continuation_{nullptr};
};

extern template class promise<void>;
} // namespace sxt::xena
