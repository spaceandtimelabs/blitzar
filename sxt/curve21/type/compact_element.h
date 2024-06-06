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

#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/zero.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// compact_element
//--------------------------------------------------------------------------------------------------
/**
 *  point (X,Y) and T = XY
 */
struct compact_element {
  f51t::element X;
  f51t::element Y;
  f51t::element T;

  static constexpr compact_element identity() noexcept {
    return compact_element{f51cn::zero_v, f51cn::one_v, f51cn::zero_v};
  }
};
} // namespace sxt::c21t
