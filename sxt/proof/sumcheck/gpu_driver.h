/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include "sxt/memory/management/managed_array_fwd.h"
#include "sxt/proof/sumcheck/driver.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// gpu_driver
//--------------------------------------------------------------------------------------------------
class gpu_driver final : public driver {
public:
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<s25t::element> mles,
                 memmg::managed_array<std::pair<s25t::element, unsigned>>&& product_table_dev,
                 memmg::managed_array<unsigned>&& product_terms_dev, unsigned n) const noexcept;

  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<s25t::element> mles,
                 basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override;

  xena::future<> sum(basct::span<s25t::element> polynomial, workspace& ws) const noexcept override;

  xena::future<> fold(workspace& ws, const s25t::element& r) const noexcept override;
};
} // namespace sxt::prfsk
