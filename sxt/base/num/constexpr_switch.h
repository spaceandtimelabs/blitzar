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
#include <utility>

#include "sxt/base/error/assert.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// constexpr_switch_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <unsigned I, unsigned First, unsigned... Rest, class F>
void constexpr_switch_impl(std::integer_sequence<unsigned, First, Rest...>, unsigned k, F f) {
  if constexpr (First < I) {
    return constexpr_switch_impl<I>(std::integer_sequence<unsigned, Rest...>{}, k, f);
  } else {
    if (First == k) {
      return f(std::integral_constant<unsigned, First>{});
    } else {
      if constexpr (sizeof...(Rest) > 0) {
        return constexpr_switch_impl<I>(std::integer_sequence<unsigned, Rest...>{}, k, f);
      } else {
        __builtin_unreachable();
      }
    }
  }
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// constexpr_switch
//--------------------------------------------------------------------------------------------------
template <unsigned I, unsigned J, class F> void constexpr_switch(unsigned k, F f) {
  // Note: use if constexpr to work around spurious compiler warnings
  if constexpr (I > 0) {
    SXT_DEBUG_ASSERT(I <= k && k < J);
  } else {
    SXT_DEBUG_ASSERT(k < J);
  }
  detail::constexpr_switch_impl<I>(std::make_integer_sequence<unsigned, J>{}, k, f);
}

template <unsigned N, class F> void constexpr_switch(unsigned k, F f) {
  return constexpr_switch<0, N>(k, f);
}
} // namespace sxt::basn
