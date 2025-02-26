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
#include "sxt/proof/sumcheck/reduction_gpu.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using s25t::operator""_s25;

TEST_CASE("we can reduce sumcheck polynomials") {
  using T = s25t::element;
  std::vector<s25t::element> p;
  std::pmr::vector<s25t::element> partial_terms{memr::get_managed_device_resource()};

  basdv::stream stream;

  SECTION("we can reduce a sum with a single term") {
    p.resize(1);
    partial_terms = {0x123_s25};
    auto fut = reduce_sums<T>(p, stream, partial_terms);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == 0x123_s25);
  }

  SECTION("we can reduce two terms") {
    p.resize(1);
    partial_terms = {0x123_s25, 0x456_s25};
    auto fut = reduce_sums<T>(p, stream, partial_terms);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == 0x123_s25 + 0x456_s25);
  }

  SECTION("we can reduce multiple coefficients") {
    p.resize(2);
    partial_terms = {0x123_s25, 0x456_s25};
    auto fut = reduce_sums<T>(p, stream, partial_terms);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == 0x123_s25);
    REQUIRE(p[1] == 0x456_s25);
  }
}
