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
#include "sxt/field32/base/montgomery.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/base/reduce.h"

using namespace sxt::f32b;

TEST_CASE("conversion to Montgomery form") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint32_t, num_limbs_v> a = {0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> ret;

    to_montgomery_form(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("with one returns one in Montgomery form") {
    constexpr std::array<uint32_t, num_limbs_v> a = {1, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> ret;

    to_montgomery_form(ret.data(), a.data());

    REQUIRE(r_v == ret);
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint32_t, num_limbs_v> expect = {0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> ret;

    to_montgomery_form(ret.data(), p_v.data());

    REQUIRE(expect == ret);
  }
}
