/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basdv {
class stream;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// copy_partial_mles
//--------------------------------------------------------------------------------------------------
void copy_partial_mles(memmg::managed_array<s25t::element>& partial_mles, basdv::stream& stream,
                       basct::cspan<s25t::element> mles, unsigned n, unsigned a,
                       unsigned b) noexcept;

//--------------------------------------------------------------------------------------------------
// copy_folded_mles
//--------------------------------------------------------------------------------------------------
void copy_folded_mles(basct::span<s25t::element> host_mles, basdv::stream& stream,
                      basct::cspan<s25t::element> device_mles, unsigned np, unsigned a,
                      unsigned b) noexcept;

//--------------------------------------------------------------------------------------------------
// get_gpu_memory_fraction
//--------------------------------------------------------------------------------------------------
double get_gpu_memory_fraction(basct::cspan<s25t::element> mles) noexcept;
} // namespace sxt::prfsk
