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
#include "sxt/curve_g1/operation/cmov.h"

#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/operation/cmov.h"

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void cmov(cg1t::element_p2& f, const cg1t::element_p2& g, unsigned int b) noexcept {
  f12o::cmov(f.X, g.X, b);
  f12o::cmov(f.Y, g.Y, b);
  f12o::cmov(f.Z, g.Z, b);
}
} // namespace sxt::cg1o
