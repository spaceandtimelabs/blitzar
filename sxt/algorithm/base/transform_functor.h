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
#include <memory_resource>
#include <utility>

#include "sxt/execution/async/future.h"

namespace sxt::basdv {
class stream;
}

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// transform_functor
//--------------------------------------------------------------------------------------------------
template <class F, class ArgFirst, class... ArgsRest>
concept transform_functor = std::copy_constructible<F> &&
                            // clang-format off
  requires(const F f, ArgFirst& x1, ArgsRest&... xrest) {
    { f(x1, xrest...) } noexcept;
  };
// clang-format on

//--------------------------------------------------------------------------------------------------
// transform_functor_factory
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class ArgFirst, class... ArgsRest> struct match_transform_functor_future {
  static constexpr bool value = false;
};

template <class F, class ArgFirst, class... ArgsRest>
  requires transform_functor<F, ArgFirst, ArgsRest...>
struct match_transform_functor_future<xena::future<F>, ArgFirst, ArgsRest...> {
  static constexpr bool value = true;
};
} // namespace detail

template <class F, class ArgFirst, class... ArgsRest>
concept transform_functor_factory =
    // clang-format off
  requires(const F f, std::pmr::polymorphic_allocator<> alloc, basdv::stream stream) {
    requires detail::match_transform_functor_future<
        decltype(f(alloc, stream)), ArgFirst, ArgsRest...>::value;
  };
// clang-format on
} // namespace sxt::algb
