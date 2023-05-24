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

#include "sxt/execution/async/future_state.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// leaf_continuation
//--------------------------------------------------------------------------------------------------
/**
 * A continuation that discards the result when its associated promise becomes available.
 */
template <class T> class leaf_continuation final : public task {
public:
  explicit leaf_continuation(future_state<T>&& state) noexcept : state_{std::move(state)} {}

  future_state<T>& state() noexcept { return state_; }

  // task
  void run_and_dispose() noexcept override { delete this; }

private:
  future_state<T> state_;
};
} // namespace sxt::xena
