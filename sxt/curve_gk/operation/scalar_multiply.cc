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
/**
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_gk/operation/scalar_multiply.h"

#include "sxt/curve_gk/operation/add.h"
#include "sxt/curve_gk/operation/double.h"
#include "sxt/curve_gk/type/element_p2.h"

namespace sxt::cgko {
//--------------------------------------------------------------------------------------------------
// is_first_bit
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static bool inline is_first_bit(const int first_byte, const int first_bit) noexcept {
  return first_byte == 31 && first_bit == 7;
}

//--------------------------------------------------------------------------------------------------
// get_first_one_bit
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static bool get_first_one_bit(int& first_one_byte, int& first_one_bit,
                              const uint8_t q[32]) noexcept {
  for (int byte_index = 31; byte_index >= 0; --byte_index) {
    auto byte = q[byte_index];
    for (int bit_index = 7; bit_index >= 0; --bit_index) {
      if (((byte >> bit_index) & 1) && !is_first_bit(byte_index, bit_index)) {
        first_one_byte = byte_index;
        first_one_bit = bit_index;
        return true;
      }
    }
  }

  first_one_byte = -1;
  first_one_bit = -1;
  return false;
}

//--------------------------------------------------------------------------------------------------
// scalar_multiply_impl
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static void scalar_multiply_impl(cgkt::element_p2& h, const cgkt::element_p2& p,
                                 const uint8_t q[32], const int first_one_byte,
                                 const int first_one_bit) noexcept {
  cgkt::element_p2 acc{cgkt::element_p2::identity()};
  int starting_bit{first_one_bit};

  for (int byte_index = first_one_byte; byte_index >= 0; --byte_index) {
    auto byte = q[byte_index];
    for (int bit_index = starting_bit; bit_index >= 0; --bit_index) {
      double_element(acc, acc);
      if ((byte >> bit_index) & 1) {
        add(acc, acc, p);
      }
    }
    starting_bit = 7; // reset starting bit for the remainder of bytes
  }

  h = acc;
}

//--------------------------------------------------------------------------------------------------
// scalar_multiply255
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void scalar_multiply255(cgkt::element_p2& h, const cgkt::element_p2& p,
                        const uint8_t q[32]) noexcept {
  int first_one_byte{0};
  int first_one_bit{0};

  if (get_first_one_bit(first_one_byte, first_one_bit, q)) {
    scalar_multiply_impl(h, p, q, first_one_byte, first_one_bit);
  } else {
    h = cgkt::element_p2::identity();
  }
}
} // namespace sxt::cgko
