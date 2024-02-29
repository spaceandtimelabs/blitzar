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

#include <cstring>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve_g1/constant/b.h"
#include "sxt/curve_g1/property/curve.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/operation/add.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/operation/sqrt.h"
#include "sxt/field12/operation/square.h"
#include "sxt/field12/random/element.h"

namespace sxt::cg1rn {
//--------------------------------------------------------------------------------------------------
// generate_random_exponent
//--------------------------------------------------------------------------------------------------
/*
 * Generates a random X,
 * solves the bls12-381 curve equation: y = sqrt(x^3 + b_v),
 * and verifies that the point is on the curve.
 */
CUDA_CALLABLE
inline void generate_random_element(cg1t::element_p2& a,
                                    basn::fast_random_number_generator& rng) noexcept {
  do {
    f12rn::generate_random_element(a.X, rng);

    f12t::element x2;
    f12t::element x3;
    f12o::square(x2, a.X);
    f12o::mul(x3, x2, a.X);

    f12t::element x3_plus_b_v;
    f12o::add(x3_plus_b_v, x3, cg1cn::b_v);

    f12o::sqrt(a.Y, x3_plus_b_v);

    a.Z = f12cn::one_v;

  } while (!cg1p::is_on_curve(a));
}
} // namespace sxt::cg1rn
