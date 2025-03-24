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

#include <cassert>
#include <utility>

#include "sxt/execution/async/future.h"
#include "sxt/execution/async/shared_future_state.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// shared_future
//--------------------------------------------------------------------------------------------------
template <class T = void> class shared_future {
public:
  shared_future() noexcept = default;

  shared_future(future<T>&& fut) noexcept {
    assert(fut.promise() != nullptr || fut.ready());
    state_ = std::make_shared<shared_future_state<T>>(std::move(fut));
  }

  future<T> make_future() const noexcept {
    assert(state_ != nullptr);
    return state_->make_future();
  }

private:
  std::shared_ptr<shared_future_state<T>> state_;
};
} // namespace sxt::xena
