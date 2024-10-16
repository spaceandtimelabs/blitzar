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

#include <array>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_reducer
//--------------------------------------------------------------------------------------------------
template <unsigned MaxDegree> struct polynomial_reducer {
  using value_type = std::array<s25t::element, MaxDegree + 1u>;

  CUDA_CALLABLE static void accumulate_inplace(value_type& res, const value_type& e) noexcept {
    for (unsigned i = 0; i < res.size(); ++i) {
      s25o::add(res[i], res[i], e[i]);
    }
  }
};
} // namespace sxt::prfsk
