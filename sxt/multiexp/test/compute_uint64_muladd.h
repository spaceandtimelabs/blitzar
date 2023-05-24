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

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// compute_uint64_muladd
//--------------------------------------------------------------------------------------------------
void compute_uint64_muladd(basct::span<uint64_t> result, basct::span<uint64_t> generators,
                           basct::span<mtxb::exponent_sequence> sequences) noexcept;
} // namespace sxt::mtxtst
