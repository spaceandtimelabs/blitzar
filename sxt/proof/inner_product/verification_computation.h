#pragma once

#include "sxt/base/container/span.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
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
