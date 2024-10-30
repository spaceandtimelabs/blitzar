/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_polynomial_01
//--------------------------------------------------------------------------------------------------
// Given a polynomial
//    f_a(X) = a[0] + a[1] * X + a[2] * X^2 + ...
// compute the sum
//    f_a(0) + f_a(1)
void sum_polynomial_01(s25t::element& e, basct::cspan<s25t::element> polynomial) noexcept;

//--------------------------------------------------------------------------------------------------
// evaluate_polynomial
//--------------------------------------------------------------------------------------------------
void evaluate_polynomial(s25t::element& e, basct::cspan<s25t::element> polynomial,
                         const s25t::element& x) noexcept;

//--------------------------------------------------------------------------------------------------
// expand_products
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void expand_products(basct::span<s25t::element> p, const s25t::element* mles, unsigned n,
                     unsigned step, basct::cspan<unsigned> terms) noexcept;

//--------------------------------------------------------------------------------------------------
// partial_expand_products
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void partial_expand_products(basct::span<s25t::element> p, const s25t::element* mles, unsigned n,
                             basct::cspan<unsigned> terms) noexcept;
} // namespace sxt::prfsk
