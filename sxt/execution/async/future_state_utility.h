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

#include <type_traits>

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/continuation_fn.h"
#include "sxt/execution/async/future_state.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// invoke_continuation_fn
//--------------------------------------------------------------------------------------------------
template <class T, class Tp, continuation_fn<T, Tp> F>
void invoke_continuation_fn(future_state<Tp>& state_p, F& f, future_state<T>& state) noexcept {
  SXT_DEBUG_ASSERT(state.ready());
  if constexpr (std::is_void_v<Tp>) {
    f(std::move(state.value()));
  } else {
    state_p.emplace(f(std::move(state.value())));
  }
}

template <class Tp, continuation_fn<void, Tp> F>
void invoke_continuation_fn(future_state<Tp>& state_p, F& f, future_state<void>& state) noexcept {
  SXT_DEBUG_ASSERT(state.ready());
  if constexpr (std::is_void_v<Tp>) {
    f();
  } else {
    state_p.emplace(f());
  }
}
} // namespace sxt::xena
