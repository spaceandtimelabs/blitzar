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

#include "sxt/field51/type/element.h"

namespace sxt::f51cn {
//--------------------------------------------------------------------------------------------------
// sqrtm1_v
//--------------------------------------------------------------------------------------------------
/* sqrt(-1) */
static constexpr f51t::element sqrtm1_v = {1718705420411056, 234908883556509, 2233514472574048,
                                           2117202627021982, 765476049583133};
} // namespace sxt::f51cn
