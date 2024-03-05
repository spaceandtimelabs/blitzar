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

#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/int.h"
#include "sxt/base/type/narrow_cast.h"

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// mac
//--------------------------------------------------------------------------------------------------
/*
 * Compute a + (b * c) + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline mac(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
  #ifdef __CUDA_ARCH__
    uint64_t mul_lo, mul_hi;
    uint64_t add_temp_lo;
    uint64_t overflow;

    // Step 1: Multiply b and c
    asm volatile(
        "mul.lo.u64 %0, %2, %3;\n\t" // Compute lower 64 bits of b*c
        "mul.hi.u64 %1, %2, %3;\n\t" // Compute upper 64 bits of b*c
        : "=l"(mul_lo), "=l"(mul_hi)
        : "l"(b), "l"(c)
    );

    // Step 2: Add a to the lower part of the product
    asm volatile(
        "add.cc.u64 %0, %2, %3;\n\t" // Compute a + mul_lo, check for carry
        "addc.u64 %1, 0, 0;\n\t"     // Add carry to 0, effectively capturing the carry
        : "=l"(add_temp_lo), "=l"(overflow)
        : "l"(a), "l"(mul_lo)
    );

    // Step 3: Add carry to the result of the previous addition, handle overflow
    asm volatile(
        "add.cc.u64 %0, %2, %3;\n\t" // Compute add_temp_lo + carry, check for carry
        "addc.u64 %1, %1, 0;\n\t"    // Add carry to the previous overflow
        : "+l"(add_temp_lo), "+l"(overflow)
        : "l"(add_temp_lo), "l"(carry)
    );

    // Step 4: Add any overflow from the multiplication to the accumulated overflow
    asm volatile(
        "add.cc.u64 %0, %1, %2;\n\t" // Add mul_hi to overflow, check for carry
        "addc.u64 %0, %0, 0;\n\t"    // Add any additional carry
        : "+l"(overflow)
        : "l"(overflow), "l"(mul_hi)
    );

    ret = add_temp_lo; // The lower 64 bits of the result
    carry = overflow;    // The upper 64 bits (including all carries and overflow)
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
 * Compute a + b + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline adc(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
//#ifdef __CUDA_ARCH__
//  asm volatile("add.u64 %0, %3, %4;\n\t"    // ret = b + c
//               "add.cc.u64 %0, %0, %2;\n\t" // ret = ret + a
//               "addc.u64 %1, 0, 0;\n\t"     // Update carry
//               : "=l"(ret), "=l"(carry)     // Set outputs
//               : "l"(a), "l"(b), "l"(c)     // Set inputs
//  );
//#else
  uint128_t ret_tmp = uint128_t{a} + uint128_t{b} + uint128_t{c};
  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
//#endif
}

//--------------------------------------------------------------------------------------------------
// sbb
//--------------------------------------------------------------------------------------------------
/*
 * Compute a - (b + borrow), returning the result and the new borrow.
 */
CUDA_CALLABLE void inline sbb(uint64_t& ret, uint64_t& borrow, const uint64_t a,
                              const uint64_t b) noexcept {
// #ifdef __CUDA_ARCH__
//   asm volatile("add.u64 %1, %3, %4;\n"             // ret = b + borrow >> 63
//                "sub.cc.u64 %0, %2, %1;\n"          // ret = ret - borrow
//                "subc.u64 %1, 0, 0;"                // Update the borrow
//                : "=l"(ret), "=l"(borrow)           // Set outputs
//                : "l"(a), "l"(b), "l"(borrow >> 63) // Set inputs
//   );
// #else
  uint128_t ret_tmp = uint128_t{a} - (uint128_t{b} + uint128_t{(borrow >> 63)});

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  borrow = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
// #endif
}

__device__ __forceinline__ uint64_t add(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm("add.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t add_cc(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t addc(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t addc_cc(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t sub(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm("sub.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t sub_cc(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t subc(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t subc_cc(const uint64_t x, const uint64_t y)
{
  uint64_t result;
  asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}
} // namespace sxt::basfld
