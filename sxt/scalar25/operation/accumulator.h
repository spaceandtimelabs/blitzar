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

#include <cstring>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/product_mapper.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// accumulator
//--------------------------------------------------------------------------------------------------
struct accumulator {
  using value_type = s25t::element;

  CUDA_CALLABLE static void accumulate_inplace(s25t::element& res, s25t::element& e,
                                               product_mapper mapper, unsigned int index) noexcept {
    e = mapper.lhs_data()[index]; // TODO: this line is subject to further benchmark assesment
    s25o::muladd(res, e, mapper.rhs_data()[index], res);
  }

  CUDA_CALLABLE static void accumulate_inplace(s25t::element& res, s25t::element& e) noexcept {
    s25o::add(res, res, e);
  }
};
} // namespace sxt::s25o
