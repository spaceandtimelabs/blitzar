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
#include "sxt/field12/base/montgomery.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/constants.h"

using namespace sxt::f12b;

TEST_CASE("we can do convert to Montgomery form as expected") {
  SECTION("one converts to one in Montgomery form") {
    constexpr std::array<uint64_t, 6> a = {1, 0, 0, 0, 0, 0};
    std::array<uint64_t, 6> ret;

    to_montgomery_form(ret.data(), a.data());

    REQUIRE(r_v == ret);
  }

  SECTION("zero converts to zero") {
    constexpr std::array<uint64_t, 6> a = {0, 0, 0, 0, 0, 0};
    constexpr std::array<uint64_t, 6> expected = {0, 0, 0, 0, 0, 0};
    std::array<uint64_t, 6> ret;

    to_montgomery_form(ret.data(), a.data());

    REQUIRE(expected == ret);
  }

  SECTION("the modulus converts to zero in Montomery form") {
    constexpr std::array<uint64_t, 6> expect = {0, 0, 0, 0, 0, 0};
    std::array<uint64_t, 6> ret;

    to_montgomery_form(ret.data(), p_v.data());

    REQUIRE(expect == ret);
  }

  SECTION("the maximum possible value converts to a pre computed value in Montomery form") {
    constexpr std::array<uint64_t, 6> a = {0xffffffffffffffff, 0xffffffffffffffff,
                                           0xffffffffffffffff, 0xffffffffffffffff,
                                           0xffffffffffffffff, 0xffffffffffffffff};
    constexpr std::array<uint64_t, 6> expected = {0x38d51f341c30c1f4, 0x3d2ee698f71904ef,
                                                  0x95cd81b5ef7f543e, 0x54947bdb16d03f3a,
                                                  0x898dcba4560e5597, 0x15a3430bd1c9e5b1};
    std::array<uint64_t, 6> ret;

    to_montgomery_form(ret.data(), a.data());

    REQUIRE(expected == ret);
  }
}
