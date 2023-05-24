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

#include <cstdint>

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// is_hex_digit
//--------------------------------------------------------------------------------------------------
constexpr bool is_hex_digit(char c) noexcept {
  if ('0' <= c && c <= '9') {
    return true;
  }
  if ('a' <= c && c <= 'f') {
    return true;
  }
  if ('A' <= c && c <= 'F') {
    return true;
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// to_hex_value
//--------------------------------------------------------------------------------------------------
constexpr uint8_t to_hex_value(char c) noexcept {
  if ('0' <= c && c <= '9') {
    return static_cast<uint8_t>(c - '0');
  }
  if ('a' <= c && c <= 'f') {
    return static_cast<uint8_t>(c - 'a' + 10);
  }
  if ('A' <= c && c <= 'F') {
    return static_cast<uint8_t>(c - 'A' + 10);
  }
  return static_cast<uint8_t>(-1);
}
} // namespace sxt::basn
