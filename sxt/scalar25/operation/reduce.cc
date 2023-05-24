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
#include "sxt/scalar25/operation/reduce.h"

#include "sxt/base/bit/load.h"
#include "sxt/scalar25/base/reduce.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// reduce33
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void reduce33(s25t::element& dest, uint8_t byte32) noexcept { s25b::reduce33(dest.data(), byte32); }

//--------------------------------------------------------------------------------------------------
// reduce64
//--------------------------------------------------------------------------------------------------
//
// Modified from libsodium's sc25519_reduce
CUDA_CALLABLE
void reduce64(s25t::element& dest, const uint8_t s[64]) noexcept {
  int64_t reduce_data[24] = {2097151LL & static_cast<int64_t>(basbt::load_3(s)),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 2) >> 5),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 5) >> 2),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 7) >> 7),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 10) >> 4),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 13) >> 1),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 15) >> 6),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 18) >> 3),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 21)),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 23) >> 5),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 26) >> 2),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 28) >> 7),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 31) >> 4),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 34) >> 1),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 36) >> 6),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 39) >> 3),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 42)),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 44) >> 5),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 47) >> 2),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 49) >> 7),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 52) >> 4),
                             2097151LL & static_cast<int64_t>(basbt::load_3(s + 55) >> 1),
                             2097151LL & static_cast<int64_t>(basbt::load_4(s + 57) >> 6),
                             static_cast<int64_t>(basbt::load_4(s + 60) >> 3)};

  s25b::reduce_impl(dest.data(), reduce_data);
}
} // namespace sxt::s25o
