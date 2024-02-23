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

namespace sxt::f32cn {
//--------------------------------------------------------------------------------------------------
// sqrtm1_v
//--------------------------------------------------------------------------------------------------
/* sqrt(-1)
 * SQRT_M1
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/constants.rs#L65-L68
 */
static constexpr f32t::element sqrtm1_v = {34513072, 25610706, 9377949,  3500415, 12389472,
                                           33281959, 41962654, 31548777, 326685,  11406482};
} // namespace sxt::f32cn
