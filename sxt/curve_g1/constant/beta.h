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

#include "sxt/field12/type/element.h"

namespace sxt::cg1cn {
//--------------------------------------------------------------------------------------------------
// beta_v
//--------------------------------------------------------------------------------------------------
/**
 * A nontrivial third root of unity in Fp
 */
static constexpr f12t::element beta_v{0x30f1361b798a64e8, 0xf3b8ddab7ece5a2a, 0x16a8ca3ac61577f7,
                                      0xc26a2ff874fd029b, 0x3636b76660701c6e, 0x051ba4ab241b6160};
} // namespace sxt::cg1cn
