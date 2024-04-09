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
/**
 * Compute a + (b * c) + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline mac(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + (uint128_t{b} * uint128_t{c}) + uint128_t{carry};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>((ret_tmp >> 64));
}

//--------------------------------------------------------------------------------------------------
// mac
//--------------------------------------------------------------------------------------------------
/**
 * Compute a + (b * c) + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline mac(uint32_t& ret, uint32_t& carry, const uint32_t a, const uint32_t b,
                              const uint32_t c) noexcept {
  uint64_t ret_tmp = uint64_t{a} + (uint64_t{b} * uint64_t{c}) + uint64_t{carry};

  ret = bast::narrow_cast<uint32_t>(ret_tmp);
  carry = bast::narrow_cast<uint32_t>((ret_tmp >> 32));
}

//--------------------------------------------------------------------------------------------------
// adc
//--------------------------------------------------------------------------------------------------
/**
 * Compute a + b + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline adc(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + uint128_t{b} + uint128_t{c};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}

//--------------------------------------------------------------------------------------------------
// adc
//--------------------------------------------------------------------------------------------------
/**
 * Compute a + b + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline adc(uint32_t& ret, uint32_t& carry, const uint32_t a, const uint32_t b,
                              const uint32_t c) noexcept {
  uint64_t ret_tmp = uint64_t{a} + uint64_t{b} + uint64_t{c};

  ret = bast::narrow_cast<uint32_t>(ret_tmp);
  carry = bast::narrow_cast<uint32_t>(ret_tmp >> 32);
}

//--------------------------------------------------------------------------------------------------
// sbb
//--------------------------------------------------------------------------------------------------
/**
 * Compute a - (b + borrow), returning the result and the new borrow.
 */
CUDA_CALLABLE void inline sbb(uint64_t& ret, uint64_t& borrow, const uint64_t a,
                              const uint64_t b) noexcept {
  uint128_t ret_tmp = uint128_t{a} - (uint128_t{b} + uint128_t{(borrow >> 63)});

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  borrow = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}

//--------------------------------------------------------------------------------------------------
// sbb
//--------------------------------------------------------------------------------------------------
/**
 * Compute a - (b + borrow), returning the result and the new borrow.
 */
CUDA_CALLABLE void inline sbb(uint32_t& ret, uint32_t& borrow, const uint32_t a,
                              const uint32_t b) noexcept {
  uint64_t ret_tmp = uint64_t{a} - (uint64_t{b} + uint64_t{(borrow >> 31)});

  ret = bast::narrow_cast<uint32_t>(ret_tmp);
  borrow = bast::narrow_cast<uint32_t>(ret_tmp >> 32);
}

namespace ptx {

__device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm("add.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t add_cc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t addc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t addc_cc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm("sub.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t sub_cc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t subc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("subc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t subc_cc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t mul_lo(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t mul_hi(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z) {
  uint32_t result;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z) {
  uint32_t result;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t mad_lo_cc(const uint32_t x, const uint32_t y,
                                              const uint32_t z) {
  uint32_t result;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t mad_hi_cc(const uint32_t x, const uint32_t y,
                                              const uint32_t z) {
  uint32_t result;
  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_lo(const uint32_t x, const uint32_t y, const uint32_t z) {
  uint32_t result;
  asm volatile("madc.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_hi(const uint32_t x, const uint32_t y, const uint32_t z) {
  uint32_t result;
  asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_lo_cc(const uint32_t x, const uint32_t y,
                                               const uint32_t z) {
  uint32_t result;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_hi_cc(const uint32_t x, const uint32_t y,
                                               const uint32_t z) {
  uint32_t result;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint64_t mov_b64(uint32_t lo, uint32_t hi) {
  uint64_t result;
  asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(lo), "r"(hi));
  return result;
}

// Gives u64 overloads a dedicated namespace.
// Callers should know exactly what they're calling (no implicit conversions).
namespace u64 {

__device__ __forceinline__ uint64_t add(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("add.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t add_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t addc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t addc_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t sub(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("sub.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t sub_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t subc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t subc_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t mul_lo(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t mul_hi(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t mad_lo(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t mad_hi(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t mad_lo_cc(const uint64_t x, const uint64_t y,
                                              const uint64_t z) {
  uint64_t result;
  asm volatile("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t mad_hi_cc(const uint64_t x, const uint64_t y,
                                              const uint64_t z) {
  uint64_t result;
  asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t madc_lo(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t madc_hi(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t madc_lo_cc(const uint64_t x, const uint64_t y,
                                               const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t madc_hi_cc(const uint64_t x, const uint64_t y,
                                               const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

} // namespace u64

__device__ __forceinline__ void bar_arrive(const unsigned name, const unsigned count) {
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(count) : "memory");
}

__device__ __forceinline__ void bar_sync(const unsigned name, const unsigned count) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(count) : "memory");
}

} // namespace ptx
} // namespace sxt::basfld
