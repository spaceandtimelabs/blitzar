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
#include <memory_resource>

#include "sxt/base/container/span.h"
#include "sxt/proof/inner_product/workspace.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
struct element;
}

namespace sxt::prfip {
struct proof_descriptor;

//--------------------------------------------------------------------------------------------------
// cpu_workspace
//--------------------------------------------------------------------------------------------------
struct cpu_workspace final : public workspace {
  std::pmr::monotonic_buffer_resource alloc;
  size_t round_index;
  const proof_descriptor* descriptor;
  basct::cspan<s25t::element> a_vector0;
  basct::span<c21t::element_p3> g_vector;
  basct::span<s25t::element> a_vector;
  basct::span<s25t::element> b_vector;

  // workspace
  xena::future<void> ap_value(s25t::element& value) const noexcept override;
};
} // namespace sxt::prfip
