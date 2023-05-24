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

#include "sxt/base/container/span.h"

namespace sxt::basct {
class blob_array;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// combine_multiproducts
//--------------------------------------------------------------------------------------------------
void combine_multiproducts(basct::span<c21t::element_p3> outputs,
                           const basct::blob_array& output_digit_or_all,
                           basct::cspan<c21t::element_p3> products) noexcept;

void combine_multiproducts(basct::span<c21t::element_p3> outputs,
                           basct::blob_array& output_digit_or_all,
                           basct::span<c21t::element_p3> products,
                           basct::cspan<mtxb::exponent_sequence> exponents) noexcept;
} // namespace sxt::mtxc21
