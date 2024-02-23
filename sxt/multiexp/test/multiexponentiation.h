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

#include <random>

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::c32t {
struct element_p3;
}
namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// multiexponentiation_fn
//--------------------------------------------------------------------------------------------------
using multiexponentiation_fn = basf::function_ref<memmg::managed_array<c32t::element_p3>(
    basct::cspan<c32t::element_p3>, basct::cspan<mtxb::exponent_sequence> exponents)>;

//--------------------------------------------------------------------------------------------------
// exercise_multiexponentiation_fn
//--------------------------------------------------------------------------------------------------
void exercise_multiexponentiation_fn(std::mt19937& rng, multiexponentiation_fn f) noexcept;
} // namespace sxt::mtxtst
