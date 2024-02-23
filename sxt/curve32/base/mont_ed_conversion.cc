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
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/curve32/base/mont_ed_conversion.h"

#include "sxt/field32/constant/one.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/invert.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/property/zero.h"

namespace sxt::c32b {
//--------------------------------------------------------------------------------------------------
// sqrtam2_v
//--------------------------------------------------------------------------------------------------
/* sqrt(-486664) */
static constexpr f32t::element ed25519_sqrtam2 = {0x3457e06, 0x1812abf, 0x350598d, 0x8a5be8,
                                                  0x316874f, 0x1fc4f7e, 0x1846e01, 0xd77a4f,
                                                  0x3460a00, 0x3c9bb7};

//--------------------------------------------------------------------------------------------------
// mont_to_ed
//--------------------------------------------------------------------------------------------------
void mont_to_ed(f32t::element& xed, f32t::element& yed, const f32t::element& x,
                const f32t::element& y) noexcept {
  f32t::element x_plus_one;
  f32t::element x_minus_one;
  f32t::element x_plus_one_y_inv;

  f32o::add(x_plus_one, x, f32cn::one_v);
  f32o::sub(x_minus_one, x, f32cn::one_v);

  /* xed = sqrt(-A-2)*x/y */
  f32o::mul(x_plus_one_y_inv, x_plus_one, y);
  f32o::invert(x_plus_one_y_inv, x_plus_one_y_inv); /* 1/((x+1)*y) */
  f32o::mul(xed, x, ed25519_sqrtam2);
  f32o::mul(xed, xed, x_plus_one_y_inv); /* sqrt(-A-2)*x/((x+1)*y) */
  f32o::mul(xed, xed, x_plus_one);

  /* yed = (x-1)/(x+1) */
  f32o::mul(yed, x_plus_one_y_inv, y); /* 1/(x+1) */
  f32o::mul(yed, yed, x_minus_one);
  f32o::cmov(yed, f32cn::one_v, f32p::is_zero(x_plus_one_y_inv));
}
} // namespace sxt::c32b
