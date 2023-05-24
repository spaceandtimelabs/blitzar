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
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#pragma once

#include <cstdint>
#include <cstring>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// load64_le
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline uint64_t load64_le(const uint8_t src[8]) noexcept {
  // note: assume the architecture is little endian
  uint64_t res;
  std::memcpy(static_cast<void*>(&res), static_cast<const void*>(src), sizeof(res));
  return res;
}

//--------------------------------------------------------------------------------------------------
// load_3
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline uint64_t load_3(const unsigned char* in) {
  uint64_t result;

  result = (uint64_t)in[0];
  result |= ((uint64_t)in[1]) << 8;
  result |= ((uint64_t)in[2]) << 16;

  return result;
}

//--------------------------------------------------------------------------------------------------
// load_4
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline uint64_t load_4(const unsigned char* in) {
  uint64_t result;

  result = (uint64_t)in[0];
  result |= ((uint64_t)in[1]) << 8;
  result |= ((uint64_t)in[2]) << 16;
  result |= ((uint64_t)in[3]) << 24;

  return result;
}
} // namespace sxt::basbt
