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

#include <array>
#include <cstdint>

namespace sxt::f12b {
//--------------------------------------------------------------------------------------------------
// p_v
//--------------------------------------------------------------------------------------------------
/*
 p_v =
 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
     =
 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
 */
static constexpr std::array<uint64_t, 6> p_v{0xb9feffffffffaaab, 0x1eabfffeb153ffff,
                                             0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                             0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
//--------------------------------------------------------------------------------------------------
// r_v
//--------------------------------------------------------------------------------------------------
/*
 r_v = 2^384 mod p
 */
static constexpr std::array<uint64_t, 6> r_v = {0x760900000002fffd, 0xebf4000bc40c0002,
                                                0x5f48985753c758ba, 0x77ce585370525745,
                                                0x5c071a97a256ec6d, 0x15f65ec3fa80e493};
//--------------------------------------------------------------------------------------------------
// r2_v
//--------------------------------------------------------------------------------------------------
/*
 r2_v = 2^(384*2) mod p
 */
static constexpr std::array<uint64_t, 6> r2_v = {0xf4df1f341c341746, 0xa76e6a609d104f1,
                                                 0x8de5476c4c95b6d5, 0x67eb88a9939d83c0,
                                                 0x9a793e85b519952d, 0x11988fe592cae3aa};
//--------------------------------------------------------------------------------------------------
// inv_v
//--------------------------------------------------------------------------------------------------
/*
 inv_v = -(p^{-1} mod 2^64) mod 2^64
 */
static constexpr uint64_t inv_v = 0x89f3fffcfffcfffd;
} // namespace sxt::f12b
