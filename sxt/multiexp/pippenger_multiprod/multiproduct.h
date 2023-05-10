/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basct {
class span_void;
}
namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxpmp {
class driver;
struct multiproduct_params;

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
void compute_multiproduct(basct::span_void inout, basct::span<basct::span<uint64_t>> products,
                          const driver& drv, size_t num_inputs) noexcept;

void compute_multiproduct(basct::span_void inout, mtxi::index_table& products, const driver& drv,
                          size_t num_inputs) noexcept;
} // namespace sxt::mtxpmp
