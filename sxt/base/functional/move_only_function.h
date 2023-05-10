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
#include <memory>
#include <type_traits>

#include "sxt/base/error/assert.h"
#include "sxt/base/functional/move_only_function_handle.h"
#include "sxt/base/functional/move_only_function_handle_impl.h"

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// move_only_function_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class, bool IsNoexcept> class move_only_function_impl;

template <class R, class... Args, bool IsNoexcept>
class move_only_function_impl<R(Args...), IsNoexcept> {
public:
  move_only_function_impl() noexcept = default;

  template <class F>
    requires requires(F f) {
      move_only_function_handle_impl<F, R(Args...), IsNoexcept>{std::move(f)};
    }
  move_only_function_impl(F&& f) noexcept
      : handle_{std::make_unique<move_only_function_handle_impl<F, R(Args...), IsNoexcept>>(
            std::move(f))} {}

  operator bool() const noexcept { return handle_ != nullptr; }

  R operator()(Args... args) noexcept(IsNoexcept) {
    SXT_DEBUG_ASSERT(handle_ != nullptr, "functor handle must be set");
    return handle_->invoke(std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<move_only_function_handle<R(Args...), IsNoexcept>> handle_;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// move_only_function
//--------------------------------------------------------------------------------------------------
/**
 * A version of std::move_only_function that works with c++20
 *
 * See https://en.cppreference.com/w/cpp/utility/functional/move_only_function
 */
template <class> class move_only_function;

template <class R, class... Args>
class move_only_function<R(Args...)> final
    : public detail::move_only_function_impl<R(Args...), false> {
public:
  using detail::move_only_function_impl<R(Args...), false>::move_only_function_impl;
};

template <class R, class... Args>
class move_only_function<R(Args...) noexcept> final
    : public detail::move_only_function_impl<R(Args...), true> {
public:
  using detail::move_only_function_impl<R(Args...), true>::move_only_function_impl;
};
} // namespace sxt::basf
