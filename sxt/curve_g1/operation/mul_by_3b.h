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

namespace sxt::f12t {
struct element;
}

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// mul_by_3b
//--------------------------------------------------------------------------------------------------
/*
 For the bls12-381 curve, since b = 4, 3b = 12.
 See Algorithm 9 for details, https://eprint.iacr.org/2015/1060.pdf
 */
CUDA_CALLABLE
void mul_by_3b(f12t::element& h, const f12t::element& p) noexcept;
} // namespace sxt::cg1o
