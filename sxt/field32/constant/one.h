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

#include "sxt/field32/base/constants.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32cn {
//--------------------------------------------------------------------------------------------------
// one_v
//--------------------------------------------------------------------------------------------------
static constexpr f32t::element one_v{f32b::r_v[0], f32b::r_v[1], f32b::r_v[2], f32b::r_v[3],
                                     f32b::r_v[4], f32b::r_v[5], f32b::r_v[6], f32b::r_v[7]};
} // namespace sxt::f32cn
