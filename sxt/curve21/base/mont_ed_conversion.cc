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
#include "sxt/curve21/base/mont_ed_conversion.h"

#include "sxt/field51/constant/one.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/invert.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/zero.h"

namespace sxt::c21b {
//--------------------------------------------------------------------------------------------------
// sqrtam2_v
//--------------------------------------------------------------------------------------------------
/* sqrt(-486664) */
static constexpr f51t::element ed25519_sqrtam2 = {
    1693982333959686, 608509411481997, 2235573344831311, 947681270984193, 266558006233600};

//--------------------------------------------------------------------------------------------------
// mont_to_ed
//--------------------------------------------------------------------------------------------------
void mont_to_ed(f51t::element& xed, f51t::element& yed, const f51t::element& x,
                const f51t::element& y) noexcept {
  f51t::element x_plus_one;
  f51t::element x_minus_one;
  f51t::element x_plus_one_y_inv;

  f51o::add(x_plus_one, x, f51cn::one_v);
  f51o::sub(x_minus_one, x, f51cn::one_v);

  /* xed = sqrt(-A-2)*x/y */
  f51o::mul(x_plus_one_y_inv, x_plus_one, y);
  f51o::invert(x_plus_one_y_inv, x_plus_one_y_inv); /* 1/((x+1)*y) */
  f51o::mul(xed, x, ed25519_sqrtam2);
  f51o::mul(xed, xed, x_plus_one_y_inv); /* sqrt(-A-2)*x/((x+1)*y) */
  f51o::mul(xed, xed, x_plus_one);

  /* yed = (x-1)/(x+1) */
  f51o::mul(yed, x_plus_one_y_inv, y); /* 1/(x+1) */
  f51o::mul(yed, yed, x_minus_one);
  f51o::cmov(yed, f51cn::one_v, f51p::is_zero(x_plus_one_y_inv));
}
} // namespace sxt::c21b
