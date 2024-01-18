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

#include "sxt/field25/type/element.h"

namespace sxt::cn1cn {
//--------------------------------------------------------------------------------------------------
// b_v
//--------------------------------------------------------------------------------------------------
/**
 * b_v is 4 in Montgomery form.
 * Used in the bls12-381 curve equation: y^2 = x^3 + 4
 */
static constexpr f12t::element b_v{0xaa270000000cfff3, 0x53cc0032fc34000a, 0x478fe97a6b0a807f,
                                   0xb1d37ebee6ba24d7, 0x8ec9733bbf78ab2f, 0x09d645513d83de7e};
} // namespace sxt::cn1cn
