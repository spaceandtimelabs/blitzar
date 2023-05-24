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

#include <array>
#include <cstddef>
#include <cstdint>

#include "sxt/base/num/hex.h"

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// parse_literal_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <size_t N, char First, char... Rest>
void parse_literal_impl(std::array<uint64_t, N>& bytes) noexcept {
  static_assert(basn::is_hex_digit(First), "invalid digit");
  auto digit_index = sizeof...(Rest);
  auto element_index = digit_index / 16;
  auto offset = (digit_index % 16) * 4;
  uint64_t val = basn::to_hex_value(First);
  bytes[element_index] += val << offset;
  if constexpr (sizeof...(Rest) > 0) {
    parse_literal_impl<N, Rest...>(bytes);
  }
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// parse_literal
//--------------------------------------------------------------------------------------------------
template <size_t N, char Z, char X, char... Digits>
void parse_literal(std::array<uint64_t, N>& bytes) noexcept {
  static_assert(Z == '0' && X == 'x' && sizeof...(Digits) > 0 && sizeof...(Digits) <= N * 16,
                "invalid field literal");
  bytes = {};
  detail::parse_literal_impl<N, Digits...>(bytes);
}
} // namespace sxt::bast
