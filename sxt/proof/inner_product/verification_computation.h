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
#pragma once

#include "sxt/base/container/span.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// compute_g_exponents
//--------------------------------------------------------------------------------------------------
void compute_g_exponents(basct::span<s25t::element> g_exponents, const s25t::element& allinv,
                         const s25t::element& ap_value,
                         basct::cspan<s25t::element> x_sq_vector) noexcept;

//--------------------------------------------------------------------------------------------------
// compute_lr_exponents_part1
//--------------------------------------------------------------------------------------------------
void compute_lr_exponents_part1(basct::span<s25t::element> l_exponents,
                                basct::span<s25t::element> r_exponents, s25t::element& allinv,
                                basct::cspan<s25t::element> x_vector) noexcept;

//--------------------------------------------------------------------------------------------------
// compute_verification_exponents
//--------------------------------------------------------------------------------------------------
/**
 * The final stage of inner product proof verification an equality check of a commitment of the
 * form
 *    <a', b'> * Q  \
 *        + s1 * g1 + ... + sn * gn \
 *        + u1 * L1 + ... + uk * Lk  \
 *        + v1 * R1 + ... + vk * Rk
 * This function computes the exponent values
 *    <a', b'>, s1, ..., sn, u1, ..., uk, v1, ..., vk
 *
 * See Protocol 2 and Section 6 from
 *  Bulletproofs: Short Proofs for Confidential Transactions and More
 *  https://www.researchgate.net/publication/326643720_Bulletproofs_Short_Proofs_for_Confidential_Transactions_and_More
 */
void compute_verification_exponents(basct::span<s25t::element> exponents,
                                    basct::cspan<s25t::element> x_vector,
                                    const s25t::element& ap_value,
                                    basct::cspan<s25t::element> b_vector) noexcept;
} // namespace sxt::prfip
