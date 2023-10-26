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

#include "sxt/scalar25/type/element.h"

namespace sxt::s25cn {
//--------------------------------------------------------------------------------------------------
// max_bits_v
//--------------------------------------------------------------------------------------------------
// Note: relying on https://en.cppreference.com/w/cpp/language/zero_initialization
static constexpr s25t::element zero_v{};
} // namespace sxt::s25cn
