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
#pragma once

#include <array>
#include <cstdint>

namespace sxt::fbnqb {
//--------------------------------------------------------------------------------------------------
// p_v
//--------------------------------------------------------------------------------------------------
/*
 p_v = 21888242871839275222246405745257275088696311157297823662689037894645226208583
     = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
 */
static constexpr std::array<uint64_t, 4> p_v{0x3c208c16d87cfd47, 0x97816a916871ca8d,
                                             0xb85045b68181585d, 0x30644e72e131a029};

//--------------------------------------------------------------------------------------------------
// r_v
//--------------------------------------------------------------------------------------------------
/*
 r_v = 2^256 mod p_v
     = 6350874878119819312338956282401532409788428879151445726012394534686998597021
     = 0xe0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d
 */
static constexpr std::array<uint64_t, 4> r_v = {0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d,
                                                0x666ea36f7879462c, 0xe0a77c19a07df2f};

//--------------------------------------------------------------------------------------------------
// r2_v
//--------------------------------------------------------------------------------------------------
/*
 r2_v = 2^(256*2) mod p_v
      = 3096616502983703923843567936837374451735540968419076528771170197431451843209
      = 0x6d89f71cab8351f47ab1eff0a417ff6b5e71911d44501fbf32cfc5b538afa89
 */
static constexpr std::array<uint64_t, 4> r2_v = {0xf32cfc5b538afa89, 0xb5e71911d44501fb,
                                                 0x47ab1eff0a417ff6, 0x6d89f71cab8351f};

//--------------------------------------------------------------------------------------------------
// inv_v
//--------------------------------------------------------------------------------------------------
/*
 inv_v = -(p_v^{-1} mod 2^64) mod 2^64
       = 9786893198990664585
       = 0x87d20782e4866389
 */
static constexpr uint64_t inv_v = 0x87d20782e4866389;
} // namespace sxt::fbnqb
