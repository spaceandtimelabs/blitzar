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
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/chained_resource.h"
#include "sxt/proof/inner_product/workspace.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
struct proof_descriptor;

static constexpr bool use_new = false;

//--------------------------------------------------------------------------------------------------
// gpu_workspace
//--------------------------------------------------------------------------------------------------
struct gpu_workspace final : public workspace {
  gpu_workspace() noexcept;

  memr::chained_resource alloc;
  size_t round_index;
  const proof_descriptor* descriptor;
  memmg::managed_array<s25t::element> a_vector;
  memmg::managed_array<s25t::element> b_vector;
  memmg::managed_array<c21t::element_p3> g_vector;

  basct::cspan<s25t::element> a_vector0X;
  basct::span<c21t::element_p3> g_vectorX;
  basct::span<s25t::element> a_vectorX;
  basct::span<s25t::element> b_vectorX;

  // workspace
  xena::future<> ap_value(s25t::element& value) const noexcept override;
};
} // namespace sxt::prfip
