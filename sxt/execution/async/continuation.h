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

#include <type_traits>

#include "sxt/execution/async/continuation_fn.h"
#include "sxt/execution/async/future_state.h"
#include "sxt/execution/async/future_state_utility.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// continuation
//--------------------------------------------------------------------------------------------------
template <class T, class Tp, continuation_fn<T, Tp> F> class continuation final : public task {
public:
  continuation(future_state<T>&& state, F&& f) noexcept
      : state_{std::move(state)}, f_{std::move(f)} {}

  continuation(const continuation&) = delete;
  continuation(continuation&&) noexcept = delete;
  continuation& operator=(const continuation&) = delete;
  continuation& operator=(continuation&&) noexcept = delete;

  future_state<T>& state() noexcept { return state_; }

  promise<Tp>& promise_p() noexcept { return promise_p_; }

  // task
  void run_and_dispose() noexcept override {
    invoke_continuation_fn(promise_p_.state(), f_, state_);
    promise_p_.make_ready();
    delete this;
  }

private:
  future_state<T> state_;
  promise<Tp> promise_p_;
  F f_;
};
} // namespace sxt::xena
