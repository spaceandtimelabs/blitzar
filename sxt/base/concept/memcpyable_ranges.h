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
#include <iterator>
#include <type_traits>
#include <utility>

namespace sxt::bascpt {
//--------------------------------------------------------------------------------------------------
// memcpyable_ranges
//--------------------------------------------------------------------------------------------------
template <class Dst, class Src, class IterDst = decltype(std::begin(std::declval<Dst>())),
          class IterSrc = decltype(std::begin(std::declval<const Src&>())),
          class T = typename std::iterator_traits<IterDst>::value_type,
          class Tp = typename std::iterator_traits<IterSrc>::value_type>
concept memcpyable_ranges =
    std::same_as<T, Tp> && std::contiguous_iterator<IterDst> && std::contiguous_iterator<IterSrc> &&
    std::output_iterator<IterDst, T> && requires(Dst dst, const Src& src) {
      // clang-format off
      { std::size(dst) } -> std::convertible_to<size_t>;
      { std::size(src) } -> std::convertible_to<size_t>;
      // clang-format on
    };
} // namespace sxt::bascpt
