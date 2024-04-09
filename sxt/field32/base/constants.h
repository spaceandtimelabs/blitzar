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
#pragma once

#include <array>
#include <cstdint>

namespace sxt::f32b {
//--------------------------------------------------------------------------------------------------
// num_limbs_v
//--------------------------------------------------------------------------------------------------
static constexpr size_t num_limbs_v = 8;

//--------------------------------------------------------------------------------------------------
// p_v
//--------------------------------------------------------------------------------------------------
/**
 * p_v = 21888242871839275222246405745257275088696311157297823662689037894645226208583
 *     = 0x30644e72 e131a029 b85045b6 8181585d 97816a91 6871ca8d 3c208c16 d87cfd47
 */
static constexpr std::array<uint32_t, num_limbs_v> p_v{
    0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};

//--------------------------------------------------------------------------------------------------
// r_v
//--------------------------------------------------------------------------------------------------
/**
 * r_v = 2^256 mod p_v
 *     = 6350874878119819312338956282401532409788428879151445726012394534686998597021
 *     = 0xe0a77c1 9a07df2f 666ea36f 7879462c 0a78eb28 f5c70b3d d35d438d c58f0d9d
 */
static constexpr std::array<uint32_t, num_limbs_v> r_v = {
    0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};

//--------------------------------------------------------------------------------------------------
// r2_v
//--------------------------------------------------------------------------------------------------
/**
 * r2_v = 2^(256*2) mod p_v
 *      = 3096616502983703923843567936837374451735540968419076528771170197431451843209
 *      = 0x6d89f71 cab8351f 47ab1eff 0a417ff6 b5e71911 d44501fb f32cfc5b 538afa89
 */
static constexpr std::array<uint32_t, num_limbs_v> r2_v = {
    0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911, 0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x6d89f71};

//--------------------------------------------------------------------------------------------------
// inv_v
//--------------------------------------------------------------------------------------------------
/**
 * inv_v = -(p_v^{-1} mod 2^32) mod 2^32
 *       = 3834012553
 *       = 0xe4866389
 */
static constexpr uint32_t inv_v = 0xe4866389;
} // namespace sxt::f32b
