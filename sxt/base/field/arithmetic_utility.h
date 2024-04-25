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
/*
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/int.h"
#include "sxt/base/type/narrow_cast.h"

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// mac
//--------------------------------------------------------------------------------------------------
/*
 Compute a + (b * c) + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline mac(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
#ifdef __CUDA_ARCH__
  uint64_t c_in = carry;
  asm volatile("{\n\t"                             // scope registers
               ".reg .u64 lo, hi;\n\t"             // create registers lo and hi
               "mul.lo.u64 lo, %2, %3;\n\t"        // lo = (b*c).lo
               "mul.hi.u64 hi, %2, %3;\n\t"        // hi = (b*c).hi
               "add.cc.u64 lo, lo, %4;\n\t"        // lo = lo + a -> CC.CF
               "addc.u64 hi, hi, 0;\n\t"           // hi = hi + CC.CF
               "add.cc.u64 lo, lo, %5;\n\t"        // lo = lo + carry -> CC.CF
               "addc.u64 hi, hi, 0;\n\t"           // hi = hi + CC.CF
               "mov.u64 %0, lo;\n\t"               // ret = lo
               "mov.u64 %1, hi;\n\t"               // carry = hi
               "}"                                 // end scope
               : "=l"(ret), "=l"(carry)            // outputs
               : "l"(b), "l"(c), "l"(a), "l"(c_in) // inputs
  );
#else
  uint128_t ret_tmp = uint128_t{a} + (uint128_t{b} * uint128_t{c}) + uint128_t{carry};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>((ret_tmp >> 64));
#endif
}

//--------------------------------------------------------------------------------------------------
// adc
//--------------------------------------------------------------------------------------------------
/*
 Compute a + b + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline adc(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
#ifdef __CUDA_ARCH__
  asm volatile("add.u64 %0, %3, %4;\n\t"    // ret = b + c
               "add.cc.u64 %0, %0, %2;\n\t" // ret = ret + a
               "addc.u64 %1, 0, 0;\n\t"     // Update carry
               : "=l"(ret), "=l"(carry)     // Set outputs
               : "l"(a), "l"(b), "l"(c)     // Set inputs
  );
#else
  uint128_t ret_tmp = uint128_t{a} + uint128_t{b} + uint128_t{c};
  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
#endif
}

//--------------------------------------------------------------------------------------------------
// sbb
//--------------------------------------------------------------------------------------------------
/*
 Compute a - (b + borrow), returning the result and the new borrow.
 */
CUDA_CALLABLE void inline sbb(uint64_t& ret, uint64_t& borrow, const uint64_t a,
                              const uint64_t b) noexcept {
#ifdef __CUDA_ARCH__
  asm volatile("add.u64 %1, %3, %4;\n"             // ret = b + borrow >> 63
              "sub.cc.u64 %0, %2, %1;\n"          // ret = ret - borrow
              "subc.u64 %1, 0, 0;"                // Update the borrow
              : "=l"(ret), "=l"(borrow)           // Set outputs
              : "l"(a), "l"(b), "l"(borrow >> 63) // Set inputs
  );
#else
  uint128_t ret_tmp = uint128_t{a} - (uint128_t{b} + uint128_t{(borrow >> 63)});

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  borrow = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
#endif
}
} // namespace sxt::basfld
