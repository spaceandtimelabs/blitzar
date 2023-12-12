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

#include "sxt/proof/inner_product/driver.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// gpu_driver
//--------------------------------------------------------------------------------------------------
class gpu_driver final : public driver {
public:
  // driver
  std::unique_ptr<workspace>
  make_workspace(const proof_descriptor& descriptor,
                 basct::cspan<s25t::element> a_vector) const noexcept override;

  xena::future<void> commit_to_fold(rstt::compressed_element& l_value,
                                    rstt::compressed_element& r_value,
                                    workspace& ws) const noexcept override;

  xena::future<void> fold(workspace& ws, const s25t::element& x) const noexcept override;

  xena::future<void>
  compute_expected_commitment(rstt::compressed_element& commit, const proof_descriptor& descriptor,
                              basct::cspan<rstt::compressed_element> l_vector,
                              basct::cspan<rstt::compressed_element> r_vector,
                              basct::cspan<s25t::element> x_vector,
                              const s25t::element& ap_value) const noexcept override;
};
} // namespace sxt::prfip
