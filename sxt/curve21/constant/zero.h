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

#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/zero.h"

namespace sxt::c21cn {
//--------------------------------------------------------------------------------------------------
// zero_p3_v
//--------------------------------------------------------------------------------------------------
static constexpr c21t::element_p3 zero_p3_v{
    .X{f51cn::zero_v},
    .Y{f51cn::one_v},
    .Z{f51cn::one_v},
    .T{f51cn::zero_v},
};

//--------------------------------------------------------------------------------------------------
// zero_cached_v
//--------------------------------------------------------------------------------------------------
static constexpr c21t::element_cached zero_cached_v{
    .YplusX{f51cn::one_v},
    .YminusX{f51cn::one_v},
    .Z{f51cn::one_v},
    .T2d{f51cn::zero_v},
};
} // namespace sxt::c21cn
