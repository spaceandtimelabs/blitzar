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

#include "sxt/field32/type/element.h"

namespace sxt::cn3cn {
//--------------------------------------------------------------------------------------------------
// b_v
//--------------------------------------------------------------------------------------------------
/**
 * b_v is 3 in Montgomery form.
 * Used in the bn254 curve equation: y^2 = x^3 + 3
 */
static constexpr f32t::element b_v{0x50ad28d7, 0x7a17caa9, 0xe15521b9, 0x1f6ac17a,
                                   0x696bd284, 0x334bea4e, 0xce179d8e, 0x2a1f6744};
} // namespace sxt::cn3cn
