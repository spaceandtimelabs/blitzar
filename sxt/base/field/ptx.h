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
 * Adopted from ingonyama-zk/icicle
 *
 * Copyright (c) 2023
 *
 * See third_party/license/ingonyama-zk.LICENSE
 */
#pragma once

#include "sxt/base/num/cmov.h"

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/**
 * result = x + y
 */
__device__ __forceinline__ uint64_t add(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("add.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// add_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = x + y -> write CC.CF
 */
__device__ __forceinline__ uint64_t add_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// addc
//--------------------------------------------------------------------------------------------------
/**
 * result = x + y + CC.CF
 */
__device__ __forceinline__ uint64_t addc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// addc_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = x + y + CC.CF -> write CC.CF
 */
__device__ __forceinline__ uint64_t addc_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
/**
 * result = x - y
 */
__device__ __forceinline__ uint64_t sub(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("sub.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// sub_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = x - y -> write CC.CF
 */
__device__ __forceinline__ uint64_t sub_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// subc
//--------------------------------------------------------------------------------------------------
/**
 * result = x - y - CC.CF
 */
__device__ __forceinline__ uint64_t subc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// subc_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = x - y - CC.CF -> write CC.CF
 */
__device__ __forceinline__ uint64_t subc_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mul_lo
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).lo
 */
__device__ __forceinline__ uint64_t mul_lo(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mul_hi
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).hi
 */
__device__ __forceinline__ uint64_t mul_hi(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mad_lo
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).lo + z
 */
__device__ __forceinline__ uint64_t mad_lo(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mad_hi
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).hi + z
 */
__device__ __forceinline__ uint64_t mad_hi(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mad_lo_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).lo + z -> write CC.CF
 */
__device__ __forceinline__ uint64_t mad_lo_cc(const uint64_t x, const uint64_t y,
                                              const uint64_t z) {
  uint64_t result;
  asm volatile("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mad_hi_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).hi + z -> write CC.CF
 */
__device__ __forceinline__ uint64_t mad_hi_cc(const uint64_t x, const uint64_t y,
                                              const uint64_t z) {
  uint64_t result;
  asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// madc_lo
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).lo + z + CC.CF
 */
__device__ __forceinline__ uint64_t madc_lo(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// madc_hi
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).hi + z + CC.CF
 */
__device__ __forceinline__ uint64_t madc_hi(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// madc_lo_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).lo + z + CC.CF -> write CC.CF
 */
__device__ __forceinline__ uint64_t madc_lo_cc(const uint64_t x, const uint64_t y,
                                               const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// madc_hi_cc
//--------------------------------------------------------------------------------------------------
/**
 * result = (x * y).hi + z + CC.CF -> write CC.CF
 */
__device__ __forceinline__ uint64_t madc_hi_cc(const uint64_t x, const uint64_t y,
                                               const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

//--------------------------------------------------------------------------------------------------
// mul_wide_limbs
//--------------------------------------------------------------------------------------------------
template <unsigned num_limbs>
__device__ __forceinline__ void mul_wide_limbs(uint64_t* c, const uint64_t* f, const uint64_t* g) noexcept {
# pragma unroll
  for (unsigned i = 0; i < num_limbs; ++i) {
    addc_cc(0,0);
# pragma unroll
    for (unsigned j = 0; j < num_limbs; ++j) {
      c[i + j] = madc_lo_cc(f[i], g[j], c[i + j]);
    }
    c[i + num_limbs] = addc_cc(0,0);
    basn::cmov(c[i + num_limbs], uint64_t{0}, i == 0);
# pragma unroll
    for (unsigned j = 0; j < num_limbs; ++j) {
      c[i + j + 1] = madc_hi_cc(f[i], g[j], c[i + j + 1]);
    }
  }
}
} // namespace sxt::basfld
