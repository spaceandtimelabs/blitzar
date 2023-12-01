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

#include "sxt/execution/async/future.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// index_functor
//--------------------------------------------------------------------------------------------------
template <class F>
concept index_functor = std::copy_constructible<F> &&
                        // clang-format off
  requires(const F f, unsigned n, unsigned i) {
    { f(n, i) } noexcept;
  };
// clang-format on

//--------------------------------------------------------------------------------------------------
// index_functor_factory
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T> struct match_index_functor_future {
  static constexpr bool value = false;
};

template <class F>
  requires index_functor<F>
struct match_index_functor_future<xena::future<F>> {
  static constexpr bool value = true;
};
} // namespace detail

template <class F>
concept index_functor_factory =
    // clang-format off
  requires(const F f, size_t a, size_t b) {
    requires detail::match_index_functor_future<decltype(f(a, b))>::value;
  };
// clang-format on
} // namespace sxt::algb
