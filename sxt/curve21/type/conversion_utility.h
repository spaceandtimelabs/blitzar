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

#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p2.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sub.h"

namespace sxt::c21t {
struct element_p3;
struct element_cached;

//--------------------------------------------------------------------------------------------------
// to_element_cached
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void to_element_cached(element_cached& r, const element_p3& p) noexcept {
  f51o::add(r.YplusX, p.Y, p.X);
  f51o::sub(r.YminusX, p.Y, p.X);
  r.Z = p.Z;
  f51o::mul(r.T2d, p.T, f51t::element{f51cn::d2_v});
}

//--------------------------------------------------------------------------------------------------
// to_element_p2
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void to_element_p2(element_p2& r, const element_p3& p) noexcept {
  r.X = p.X;
  r.Y = p.Y;
  r.Z = p.Z;
}

CUDA_CALLABLE
inline void to_element_p2(element_p2& r, const element_p1p1& p) noexcept {
  f51o::mul(r.X, p.X, p.T);
  f51o::mul(r.Y, p.Y, p.Z);
  f51o::mul(r.Z, p.Z, p.T);
}

//--------------------------------------------------------------------------------------------------
// to_element_p3
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T>
CUDA_CALLABLE inline void to_element_p3_impl(T& r, const element_p1p1& p) noexcept {
  f51o::mul(r.X, p.X, p.T);
  f51o::mul(r.Y, p.Y, p.Z);
  f51o::mul(r.Z, p.Z, p.T);
  f51o::mul(r.T, p.X, p.Y);
}

} // namespace detail

CUDA_CALLABLE
inline void to_element_p3(element_p3& r, const element_p1p1& p) noexcept {
  detail::to_element_p3_impl(r, p);
}
} // namespace sxt::c21t
