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

#include <list>
#include <memory>
#include <utility>

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// shared_future_state
//--------------------------------------------------------------------------------------------------
/**
 * Manage state for a future that can be awaited multiple times.
 *
 * This is a highly simplified version of a shared_future derived from seastar
 *
 * See https://seastar.io/futures-promises/
 */
template <class T>
class shared_future_state final : public task,
                                  public std::enable_shared_from_this<shared_future_state<T>> {
public:
  shared_future_state() noexcept = default;
  shared_future_state(future<T>&& fut) noexcept : fut_{std::move(fut)} {
    SXT_DEBUG_ASSERT(fut_.promise() != nullptr || fut_.ready());
  }

  shared_future_state(const shared_future_state&) = delete;
  shared_future_state(shared_future_state&&) = delete;

  shared_future_state& operator=(const shared_future_state&) = delete;
  shared_future_state& operator=(shared_future_state&&) = delete;

  future<T> make_future() noexcept {
    if (fut_.ready()) {
      if constexpr (std::is_same_v<T, void>) {
        return make_ready_future();
      } else {
        return make_ready_future<T>(T{fut_.value()});
      }
    }
    if (promises_.empty()) {
      fut_.promise()->set_continuation(*this);
      keep_alive_ = this->shared_from_this();
    }
    promises_.emplace_back();
    return future<T>{promises_.back()};
  };

private:
  void run_and_dispose() noexcept override {
    while (!promises_.empty()) {
      if constexpr (std::is_same_v<T, void>) {
        promises_.back().make_ready();
      } else {
        promises_.back().set_value(fut_.value());
      }
      promises_.pop_back();
    }
    keep_alive_.reset();
  }

  std::shared_ptr<shared_future_state<T>> keep_alive_;
  future<T> fut_;
  std::list<promise<T>> promises_;
};
} // namespace sxt::xena
