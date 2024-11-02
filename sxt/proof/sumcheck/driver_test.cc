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
#include "sxt/proof/sumcheck/driver_test.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/sumcheck/driver.h"
#include "sxt/proof/sumcheck/workspace.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
using s25t::operator""_s25;

//--------------------------------------------------------------------------------------------------
// exercise_driver
//--------------------------------------------------------------------------------------------------
void exercise_driver(const driver& drv) {
  std::vector<s25t::element> mles;
  std::vector<std::pair<s25t::element, unsigned>> product_table{
      {0x1_s25, 1},
  };
  std::vector<unsigned> product_terms = {0};

  std::vector<s25t::element> p(2);

  SECTION("we can sum a polynomial with n = 1") {
    std::vector<s25t::element> mles = {0x123_s25};
    auto ws = drv.make_workspace(mles, product_table, product_terms, 1).value();
    auto fut = drv.sum(p, *ws);
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == -mles[0]);
  }

  SECTION("we can sum a polynomial with a non-unity multiplier") {
    std::vector<s25t::element> mles = {0x123_s25};
    product_table[0].first = 0x2_s25;
    auto ws = drv.make_workspace(mles, product_table, product_terms, 1).value();
    auto fut = drv.sum(p, *ws);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == 0x2_s25 * mles[0]);
    REQUIRE(p[1] == -0x2_s25 * mles[0]);
  }

  SECTION("we can sum a polynomial with n = 2") {
    std::vector<s25t::element> mles = {0x123_s25, 0x456_s25};
    auto ws = drv.make_workspace(mles, product_table, product_terms, 2).value();
    auto fut = drv.sum(p, *ws);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }

  SECTION("we can sum a polynomial with two MLEs added together") {
    std::vector<s25t::element> mles = {0x123_s25, 0x456_s25};
    std::vector<std::pair<s25t::element, unsigned>> product_table{
        {0x1_s25, 1},
        {0x1_s25, 1},
    };
    std::vector<unsigned> product_terms = {0, 1};

    auto ws = drv.make_workspace(mles, product_table, product_terms, 1).value();
    auto fut = drv.sum(p, *ws);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == -mles[0] - mles[1]);
  }

  SECTION("we can sum a polynomial with two MLEs multiplied together") {
    std::vector<s25t::element> mles = {0x123_s25, 0x456_s25};
    std::vector<std::pair<s25t::element, unsigned>> product_table{
        {0x1_s25, 2},
    };
    std::vector<unsigned> product_terms = {0, 1};
    p.resize(3);

    auto ws = drv.make_workspace(mles, product_table, product_terms, 1).value();
    auto fut = drv.sum(p, *ws);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] * mles[1]);
    REQUIRE(p[1] == -mles[0] * mles[1] - mles[1] * mles[0]);
    REQUIRE(p[2] == mles[0] * mles[1]);
  }

  SECTION("we can fold mles") {
    std::vector<s25t::element> mles = {0x123_s25, 0x456_s25, 0x789_s25};
    auto ws = drv.make_workspace(mles, product_table, product_terms, 3).value();
    auto r = 0xabc123_s25;
    auto fut = drv.fold(*ws, r);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    fut = drv.sum(p, *ws);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    mles[0] = (0x1_s25 - r) * mles[0] + r * mles[2];
    mles[1] = (0x1_s25 - r) * mles[1];

    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }
}
} // namespace sxt::prfsk
