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

#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
struct element;
}
namespace sxt::rstt {
class compressed_element;
}

namespace sxt::prfip {
class workspace;
struct proof_descriptor;

//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
/**
 * Provide interfaces for the computationally intensive parts of proving and verifying inner
 * products.
 *
 * This abstraction allows the same top-level proof code to use different computational backends.
 */
class driver {
public:
  virtual ~driver() noexcept = default;

  /**
   * Create a workspace that persists through the proving of an inner product and can be used to
   * store context that's referenced by multiple rounds of the proving.
   */
  virtual xena::future<std::unique_ptr<workspace>>
  make_workspace(const proof_descriptor& descriptor,
                 basct::cspan<s25t::element> a_vector) const noexcept = 0;

  /**
   * Commit to the L-R split of an inner product problem.
   *
   * See Protocol 2, Lines 23-24 from
   *  Bulletproofs: Short Proofs for Confidential Transactions and More
   *  https://www.researchgate.net/publication/326643720_Bulletproofs_Short_Proofs_for_Confidential_Transactions_and_More
   */
  virtual xena::future<void> commit_to_fold(rstt::compressed_element& l_value,
                                            rstt::compressed_element& r_value,
                                            workspace& ws) const noexcept = 0;

  /**
   * Using the randomly selected scalar x, fold an inner product proof to achieve a proof
   * of half the size.
   *
   * See Protocol 2, Lines 28-34 from
   *  Bulletproofs: Short Proofs for Confidential Transactions and More
   *  https://www.researchgate.net/publication/326643720_Bulletproofs_Short_Proofs_for_Confidential_Transactions_and_More
   */
  virtual xena::future<void> fold(workspace& ws, const s25t::element& x) const noexcept = 0;

  /**
   * Compute the expected commitment of an inner product proof after it's been repeatedly folded
   * down to a proof with a single element.
   *
   * See Equation 4 from
   *  Bulletproofs: Short Proofs for Confidential Transactions and More
   *  https://www.researchgate.net/publication/326643720_Bulletproofs_Short_Proofs_for_Confidential_Transactions_and_More
   */
  virtual xena::future<void>
  compute_expected_commitment(rstt::compressed_element& commit, const proof_descriptor& descriptor,
                              basct::cspan<rstt::compressed_element> l_vector,
                              basct::cspan<rstt::compressed_element> r_vector,
                              basct::cspan<s25t::element> x_vector,
                              const s25t::element& ap_value) const noexcept = 0;
};
} // namespace sxt::prfip
