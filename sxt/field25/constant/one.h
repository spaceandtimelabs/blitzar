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

#include "sxt/field12/base/constants.h"
#include "sxt/field12/type/element.h"

namespace sxt::f12cn {
//--------------------------------------------------------------------------------------------------
// one_v
//--------------------------------------------------------------------------------------------------
static constexpr f12t::element one_v{f12b::r_v[0], f12b::r_v[1], f12b::r_v[2],
                                     f12b::r_v[3], f12b::r_v[4], f12b::r_v[5]};
} // namespace sxt::f12cn
