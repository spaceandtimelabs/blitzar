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

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::fgkb {
//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
/**
 * h = s mod p
 * If s represents a above the modulus, the from_bytes function will set the is_below_modulus flag
 * to false and return a wrapped the value, h. In this case s != to_bytes(h). Otherwise the
 * is_below_modulus flag will be set to true and s == to_bytes(h).
 */
CUDA_CALLABLE
void from_bytes(bool& is_below_modulus, uint64_t h[4], const uint8_t s[32]) noexcept;

//--------------------------------------------------------------------------------------------------
// from_bytes_le
//--------------------------------------------------------------------------------------------------
/**
 * The internal representation of fieldgk elements are in little-endian order.
 */
CUDA_CALLABLE
void from_bytes_le(bool& is_below_modulus, uint64_t h[4], const uint8_t s[32]) noexcept;

//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[32], const uint64_t h[4]) noexcept;

//--------------------------------------------------------------------------------------------------
// to_bytes_le
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes_le(uint8_t s[32], const uint64_t h[4]) noexcept;
} // namespace sxt::fgkb
