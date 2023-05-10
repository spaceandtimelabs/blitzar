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

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// is_continuation_fn_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class F, class From, class T> struct is_continuation_fn_impl {
  static constexpr bool value = false;
};

template <class F, class From, class To>
  requires requires(F f, From from) {
    { f(std::move(from)) } noexcept -> std::convertible_to<To>;
  }
struct is_continuation_fn_impl<F, From, To> {
  static constexpr bool value = true;
};

template <class F, class To>
  requires requires(F f) {
    { f() } noexcept -> std::convertible_to<To>;
  }
struct is_continuation_fn_impl<F, void, To> {
  static constexpr bool value = true;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// continuation_fn
//--------------------------------------------------------------------------------------------------
template <class F, class From, class To>
concept continuation_fn = detail::is_continuation_fn_impl<F, From, To>::value;
} // namespace sxt::xena
