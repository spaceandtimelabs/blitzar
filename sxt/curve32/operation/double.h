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
#include "sxt/curve32/type/conversion_utility.h"
#include "sxt/curve32/type/double_impl.h"
#include "sxt/curve32/type/element_p1p1.h"
#include "sxt/curve32/type/element_p2.h"

namespace sxt::c32o {
//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
/*
 r = 2 * p
*/
CUDA_CALLABLE
inline void double_element(c32t::element_p1p1& r, const c32t::element_p2& p) noexcept {
  c32t::double_element_impl(r, p);
}

CUDA_CALLABLE
inline void double_element(c32t::element_p1p1& r, const c32t::element_p3& p) noexcept {
  c32t::double_element_impl(r, p);
}

CUDA_CALLABLE
inline void double_element(c32t::element_p3& r, const c32t::element_p3& p) noexcept {
  c32t::element_p1p1 t;
  double_element(t, p);
  to_element_p3(r, t);
}
} // namespace sxt::c32o
