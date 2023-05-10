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
#include "sxt/ristretto/base/point_formation.h"

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/type/element.h"
#include "sxt/ristretto/base/elligator.h"

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// form_ristretto_point
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void form_ristretto_point(c21t::element_p3& p, const f51t::element& r0,
                          const f51t::element& r1) noexcept {
  c21t::element_p3 p0;
  apply_elligator(p0, r0);
  apply_elligator(p, r1);
  c21o::add(p, p, p0);
}
} // namespace sxt::rstb
