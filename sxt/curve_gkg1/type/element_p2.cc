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
#include "sxt/curve_gkg1/type/element_p2.h"

#include "sxt/fieldgk/operation/invert.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/property/zero.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::ck1t {
//--------------------------------------------------------------------------------------------------
// unset_marker_v
//--------------------------------------------------------------------------------------------------
static constexpr uint64_t unset_marker_v = static_cast<uint64_t>(-1);

//--------------------------------------------------------------------------------------------------
// mark
//--------------------------------------------------------------------------------------------------
void mark(element_p2& e) noexcept { e.Z[3] = unset_marker_v; }

//--------------------------------------------------------------------------------------------------
// is_marked
//--------------------------------------------------------------------------------------------------
bool is_marked(const element_p2& e) noexcept { return e.Z[3] != unset_marker_v; }

//--------------------------------------------------------------------------------------------------
// operator compact_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE element_p2::operator compact_element() const noexcept {
  fgkt::element z_inv;
  const bool is_zero{fgko::invert(z_inv, Z)};
  fgko::cmov(z_inv, fgkcn::zero_v, is_zero);

  fgkt::element x;
  fgkt::element y;
  fgko::mul(x, X, z_inv);
  fgko::mul(y, Y, z_inv);

  fgko::cmov(x, compact_element::identity().X, is_zero);
  fgko::cmov(y, compact_element::identity().Y, is_zero);

  return {x, y};
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if either both points are at infinity or neither point is at infinity,
 * and the coordinates are the same.
 */
bool operator==(const element_p2& lhs, const element_p2& rhs) noexcept {
  fgkt::element x1;
  fgkt::element x2;
  fgkt::element y1;
  fgkt::element y2;

  fgko::mul(x1, lhs.X, rhs.Z);
  fgko::mul(x2, rhs.X, lhs.Z);

  fgko::mul(y1, lhs.Y, rhs.Z);
  fgko::mul(y2, rhs.Y, lhs.Z);

  const auto lhs_is_zero = fgkp::is_zero(lhs.Z);
  const auto rhs_is_zero = fgkp::is_zero(rhs.Z);

  return (lhs_is_zero && rhs_is_zero) ||
         ((!lhs_is_zero) && (!rhs_is_zero) && (x1 == x2) && (y1 == y2));
}
} // namespace sxt::ck1t
