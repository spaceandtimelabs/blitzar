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
#include "sxt/proof/inner_product/scalar_fold_kernel.h"

#include <vector>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;
using s25t::operator""_s25;

TEST_CASE("we can fold scalars") {
  std::vector<s25t::element> scalars;

  auto m_low = 0x234987_s25;
  auto m_high = 0x986798234_s25;

  SECTION("we can fold two scalars") {
    scalars = {0x123_s25, 0x456_s25};
    auto fut = async_fold_scalars(basct::subspan(scalars, 0, 1), scalars, m_low, m_high);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(scalars[0] == 0x123_s25 * m_low + 0x456_s25 * m_high);
  }

  SECTION("we can fold three scalars") {
    scalars = {0x123_s25, 0x456_s25, 0x789_s25};
    auto fut = async_fold_scalars(basct::subspan(scalars, 0, 2), scalars, m_low, m_high);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(scalars[0] == 0x123_s25 * m_low + 0x789_s25 * m_high);
    REQUIRE(scalars[1] == 0x456_s25 * m_low);
  }
}
