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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// accumulator
//--------------------------------------------------------------------------------------------------
struct accumulator {
  using value_type = c21t::element_p3;

  CUDA_CALLABLE static inline void accumulate_inplace(c21t::element_p3& p,
                                                     c21t::element_p3& q) noexcept {
    f51o::add(q.X, q.Y, q.X);
    f51o::add(q.Y, q.Y, q.Y);
    f51o::sub(q.Y, q.Y, q.X);
    f51o::mul(q.T, q.T, f51t::element{f51cn::d2_v});

    // add p and q_cached
    f51o::add(p.X, p.Y, p.X);
    f51o::add(p.Y, p.Y, p.Y);
    f51o::sub(p.Y, p.Y, p.X);
    f51o::mul(p.Y, p.Y, q.Y);
    f51o::mul(p.T, q.T, p.T);
    f51o::mul(p.X, p.X, q.X);
    f51o::sub(q.X, p.X, p.Y);
    f51o::add(q.Y, p.X, p.Y);
    f51o::mul(p.Z, p.Z, q.Z);
    f51o::add(q.T, p.Z, p.Z);
    f51o::add(q.Z, q.T, p.T);
    f51o::sub(q.T, q.T, p.T);

    // convert q back into a `c21t::element_p3`
    f51o::mul(p.X, q.X, q.T);
    f51o::mul(p.Y, q.Y, q.Z);
    f51o::mul(p.Z, q.Z, q.T);
    f51o::mul(p.T, q.X, q.Y);
  }
};
} // namespace sxt::c21o
