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

namespace sxt::fgkb {
//--------------------------------------------------------------------------------------------------
// p_v
//--------------------------------------------------------------------------------------------------
/**
 * p_v = 21888242871839275222246405745257275088548364400416034343698204186575808495617
 *     = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
 */
static constexpr std::array<uint64_t, 4> p_v{0x43e1f593f0000001, 0x2833e84879b97091,
                                             0xb85045b68181585d, 0x30644e72e131a029};

//--------------------------------------------------------------------------------------------------
// r_v
//--------------------------------------------------------------------------------------------------
/**
 * r_v = 2^256 mod p_v
 *     = 6350874878119819312338956282401532410528162663560392320966563075034087161851
 *     = 0xe0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb
 */
static constexpr std::array<uint64_t, 4> r_v = {0xac96341c4ffffffb, 0x36fc76959f60cd29,
                                                0x666ea36f7879462e, 0x0e0a77c19a07df2f};

//--------------------------------------------------------------------------------------------------
// r2_v
//--------------------------------------------------------------------------------------------------
/**
 * r2_v = 2^(256*2) mod p_v
 *      = 944936681149208446651664254269745548490766851729442924617792859073125903783
 *      = 0x216d0b17f4e44a58c49833d53bb808553fe3ab1e35c59e31bb8e645ae216da7
 */
static constexpr std::array<uint64_t, 4> r2_v = {0x1bb8e645ae216da7, 0x53fe3ab1e35c59e3,
                                                 0x8c49833d53bb8085, 0x0216d0b17f4e44a5};

//--------------------------------------------------------------------------------------------------
// inv_v
//--------------------------------------------------------------------------------------------------
/**
 * inv_v = -(p_v^{-1} mod 2^64) mod 2^64
 *       = 14042775128853446655
 *       = 0xc2e1f593efffffff
 */
static constexpr uint64_t inv_v = 0xc2e1f593efffffff;
} // namespace sxt::fgkb
