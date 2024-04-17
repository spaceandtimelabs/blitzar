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

namespace sxt::basfld {
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
} // namespace sxt::basfld
