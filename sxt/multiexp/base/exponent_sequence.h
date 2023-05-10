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

#include <cstdint>

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// exponent_sequence
//--------------------------------------------------------------------------------------------------
struct exponent_sequence {
  // the number of bytes used to represent an element in the sequence
  // element_nbytes must be a power of 2 and must satisfy
  //    1 <= element_nbytes <= 32
  uint8_t element_nbytes = 0;

  // the number of elements in the sequence
  uint64_t n = 0;

  // pointer to the data for the sequence of elements where there are n elements
  // in the sequence and each element encodes a number of element_nbytes bytes
  // represented in the little endian format
  const uint8_t* data = nullptr;

  // whether the elements are signed
  // Note: if signed, then element_nbytes must be <= 16
  int is_signed = 0;
};
} // namespace sxt::mtxb
