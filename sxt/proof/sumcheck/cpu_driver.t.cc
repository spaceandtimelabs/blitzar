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
#include "sxt/proof/sumcheck/cpu_driver.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/proof/sumcheck/workspace.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can perform the primitive operations for sumcheck proofs") {
  std::vector<s25t::element> mles = {0x123_s25};
  std::vector<std::pair<s25t::element, unsigned>> product_table{
      {0x1_s25, 1},
  };
  std::vector<unsigned> product_terms = {0};

  cpu_driver drv;
  auto ws = drv.make_workspace(mles, product_table, product_terms, 1).value();
  (void)ws;
#if 0
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<s25t::element> mles,
                 basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override;

  xena::future<> sum(basct::span<s25t::element> polynomial, workspace& ws) const noexcept override;

  xena::future<> fold(workspace& ws, const s25t::element& r) const noexcept override;
#endif
}
