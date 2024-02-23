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

#include "sxt/curve32/type/element_cached.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/zero.h"

namespace sxt::c32cn {
//--------------------------------------------------------------------------------------------------
// identity_cached_v
//--------------------------------------------------------------------------------------------------
static constexpr c32t::element_cached identity_cached_v{
    .YplusX{f32cn::one_v},
    .YminusX{f32cn::one_v},
    .Z{f32cn::one_v},
    .T2d{f32cn::zero_v},
};
} // namespace sxt::c32cn
