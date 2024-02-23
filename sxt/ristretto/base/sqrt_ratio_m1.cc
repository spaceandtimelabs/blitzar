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

#include "sxt/ristretto/base/sqrt_ratio_m1.h"

#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/constant/sqrtm1.h"
#include "sxt/field32/operation/abs.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/pow22523.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/property/zero.h"

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// compute_sqrt_ratio_m1
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int compute_sqrt_ratio_m1(f32t::element& x, const f32t::element& u,
                          const f32t::element& v) noexcept {
  f32t::element v3;
  f32t::element vxx;
  f32t::element m_root_check, p_root_check, f_root_check;
  f32t::element x_sqrtm1;
  int has_m_root, has_p_root, has_f_root;

  f32o::sq(v3, v);
  f32o::mul(v3, v3, v); /* v3 = v^3 */
  f32o::sq(x, v3);
  f32o::mul(x, x, u);
  f32o::mul(x, x, v); /* x = uv^7 */

  f32o::pow22523(x, x); /* x = (uv^7)^((q-5)/8) */
  f32o::mul(x, x, v3);
  f32o::mul(x, x, u); /* x = uv^3(uv^7)^((q-5)/8) */

  f32o::sq(vxx, x);
  f32o::mul(vxx, vxx, v);                                     /* vx^2 */
  f32o::sub(m_root_check, vxx, u);                            /* vx^2-u */
  f32o::add(p_root_check, vxx, u);                            /* vx^2+u */
  f32o::mul(f_root_check, u, f32t::element{f32cn::sqrtm1_v}); /* u*sqrt(-1) */
  f32o::add(f_root_check, vxx, f_root_check);                 /* vx^2+u*sqrt(-1) */
  has_m_root = f32p::is_zero(m_root_check);
  has_p_root = f32p::is_zero(p_root_check);
  has_f_root = f32p::is_zero(f_root_check);
  f32o::mul(x_sqrtm1, x, f32t::element{f32cn::sqrtm1_v}); /* x*sqrt(-1) */

  f32o::cmov(x, x_sqrtm1, has_p_root | has_f_root);
  f32o::abs(x, x);

  return has_m_root | has_p_root;
}
} // namespace sxt::rstb
