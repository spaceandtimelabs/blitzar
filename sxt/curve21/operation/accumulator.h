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
#include "sxt/curve21/type/element_p3.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// accumulator
//--------------------------------------------------------------------------------------------------
struct accumulator {
  using value_type = c21t::element_p3;

  CUDA_CALLABLE static void accumulate_inplace(c21t::element_p3& res,
                                               c21t::element_p3& e) noexcept {
    add_inplace(res, e);
  }
};
} // namespace sxt::c21o
