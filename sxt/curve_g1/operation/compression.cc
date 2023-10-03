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
#include "sxt/curve_g1/operation/compression.h"

#include "sxt/base/error/assert.h"
#include "sxt/base/num/cmov.h"
#include "sxt/curve_g1/type/compressed_element.h"
#include "sxt/curve_g1/type/conversion_utility.h"
#include "sxt/curve_g1/type/element_affine.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/base/byte_conversion.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/operation/cmov.h"
#include "sxt/field12/property/lexicographically_largest.h"

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// compress
//--------------------------------------------------------------------------------------------------
void compress(cg1t::compressed_element& e_c, const cg1t::element_p2& e_p) noexcept {
  cg1t::element_affine e_a;
  cg1t::to_element_affine(e_a, e_p);

  f12o::cmov(e_a.X, f12cn::zero_v, e_a.infinity);

  f12b::to_bytes(e_c.data(), e_a.X.data());

  // This point is in compressed form, so we set the most significant bit.
  e_c.data()[47] |= static_cast<uint8_t>(1) << 7;

  // Is this point at infinity? If so, set the second-most significant bit.
  uint8_t pt_inf{static_cast<uint8_t>(0)};
  constexpr uint8_t pt_inf_bit{static_cast<uint8_t>(1) << 6};
  basn::cmov(pt_inf, pt_inf_bit, e_a.infinity);
  e_c.data()[47] |= pt_inf;

  // Is the y-coordinate the lexicographically largest of the two associated with the
  // x-coordinate? If so, set the third-most significant bit so long as this is not
  // the point at infinity.
  uint8_t y_lx_lrg{static_cast<uint8_t>(0)};
  constexpr uint8_t lx_lrg_bit{static_cast<uint8_t>(1) << 5};
  const bool t{!e_a.infinity && f12p::lexicographically_largest(e_a.Y)};
  basn::cmov(y_lx_lrg, lx_lrg_bit, t);
  e_c.data()[47] |= y_lx_lrg;
}

//--------------------------------------------------------------------------------------------------
// batch_compress
//--------------------------------------------------------------------------------------------------
void batch_compress(basct::span<cg1t::compressed_element> ex_c,
                    basct::cspan<cg1t::element_p2> ex_p) noexcept {
  SXT_DEBUG_ASSERT(ex_c.size() == ex_p.size());
  for (size_t i = 0; i < ex_p.size(); ++i) {
    compress(ex_c[i], ex_p[i]);
  }
}
} // namespace sxt::cg1o
