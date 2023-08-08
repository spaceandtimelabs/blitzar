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

#include "sxt/curve_g1/type/element_affine.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/type/element.h"

namespace sxt::cg1cn {
//--------------------------------------------------------------------------------------------------
// identity_affine_v
//--------------------------------------------------------------------------------------------------
static constexpr cg1t::element_affine identity_affine_v{f12cn::zero_v, f12cn::one_v, true};
} // namespace sxt::cg1cn
