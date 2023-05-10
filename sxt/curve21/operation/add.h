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

#include <concepts>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/d.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(c21t::element_p1p1& r, const c21t::element_p3& p, const c21t::element_cached& q) noexcept;

CUDA_CALLABLE
inline void add(c21t::element_p3& res, const c21t::element_p3& lhs,
                const c21t::element_p3& rhs) noexcept {
  c21t::element_cached t;
  to_element_cached(t, rhs);
  c21t::element_p1p1 res_p;
  add(res_p, lhs, t);
  to_element_p3(res, res_p);
}

//--------------------------------------------------------------------------------------------------
// add_inplace
//--------------------------------------------------------------------------------------------------
/*
 p = p + q (q value is not preserved)
 */
template <class T>
  requires std::same_as<T, c21t::element_p3> || std::same_as<T, volatile c21t::element_p3>
CUDA_CALLABLE inline void add_inplace(T& p, T& q) noexcept {
  // convert q into a `c21t::element_cached`
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
} // namespace sxt::c21o
