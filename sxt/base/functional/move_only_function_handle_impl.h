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
#include <functional>
#include <type_traits>

#include "sxt/base/functional/move_only_function_handle.h"

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// move_only_function_handle_impl
//--------------------------------------------------------------------------------------------------
template <class, class, bool> class move_only_function_handle_impl;

template <class F, class R, class... Args, bool IsNoexcept>
  requires std::is_invocable_r_v<R, F, Args...>
class move_only_function_handle_impl<F, R(Args...), IsNoexcept> final
    : public move_only_function_handle<R(Args...), IsNoexcept> {
public:
  explicit move_only_function_handle_impl(F&& f) noexcept : f_{std::move(f)} {}

  R invoke(Args... args) noexcept(IsNoexcept) override {
    return std::invoke(f_, std::forward<Args>(args)...);
  }

private:
  F f_;
};
} // namespace sxt::basf
