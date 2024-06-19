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

namespace sxt::ck1cn {
//--------------------------------------------------------------------------------------------------
// b_v
//--------------------------------------------------------------------------------------------------
/**
 * b_v is 3 in Montgomery form.
 * Used in the bn254 curve equation: y^2 = x^3 + 3
 */
static constexpr f25t::element b_v{0x7a17caa950ad28d7, 0x1f6ac17ae15521b9, 0x334bea4e696bd284,
                                   0x2a1f6744ce179d8e};
} // namespace sxt::ck1cn
