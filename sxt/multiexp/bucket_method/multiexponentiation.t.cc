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
#include "sxt/multiexp/bucket_method/multiexponentiation.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/field51/type/literal.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk;
using c21t::operator""_c21;
using f51t::operator""_f51;

TEST_CASE("we can compute a multiexponentiation") {
  std::vector<bascrv::element97> res(1);
  std::vector<bascrv::element97> generators;
  std::vector<const uint8_t*> exponents;

  SECTION("we can compute a multiexponentiation with no elements") {
    res.clear();
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    REQUIRE(fut.ready());
  }

  SECTION("we can compute a multiexponentiation with a single zero element") {
    uint8_t scalar_data[32] = {};
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 0u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 1") {
    uint8_t scalar_data[32] = {};
    scalar_data[0] = 1;
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 12u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 2") {
    uint8_t scalar_data[32] = {};
    scalar_data[0] = 2;
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 24u);
  }

  SECTION("we can compute a multiexponentiation with a single element of 256") {
    uint8_t scalar_data[32] = {};
    scalar_data[1] = 1;
    exponents.push_back(scalar_data);
    generators = {12u};
    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 256u * 12u);
  }

  SECTION("we can compute a multiexponentiation with 2 generators") {
    uint8_t scalar_data[64] = {};
    scalar_data[0] = 2;
    scalar_data[32] = 3;
    exponents.push_back(scalar_data);

    generators = {12u, 34u};

    auto fut = multiexponentiate<bascrv::element97>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * 12u + 3u * 34u);
  }
}

TEST_CASE("we can compute multiexponentiations with curve-21") {
  std::vector<c21t::element_p3> res(1);
  std::vector<c21t::element_p3> generators;
  std::vector<const uint8_t*> exponents;

  SECTION("we can compute a multiexponentiation with a single element of 1") {
    uint8_t scalar_data[32] = {};
    scalar_data[0] = 1;
    exponents.push_back(scalar_data);
    generators = {0x123_c21};
    auto fut = multiexponentiate<c21t::element_p3>(res, generators, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 0x123_c21);
  }
}