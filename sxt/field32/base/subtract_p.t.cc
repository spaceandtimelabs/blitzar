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
#include "sxt/field32/base/subtract_p.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/constants.h"

using namespace sxt::f32b;

TEST_CASE("subtract_p (subtraction with the modulus) can handle computation") {
  SECTION("of zero") {
    constexpr std::array<uint32_t, num_limbs_v> a = {0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("of one below the modulus p_v") {
    constexpr std::array<uint32_t, num_limbs_v> a = {0xd87cfd46, 0x3c208c16, 0x6871ca8d,
                                                     0x97816a91, 0x8181585d, 0xb85045b6,
                                                     0xe131a029, 0x30644e72};
    std::array<uint32_t, num_limbs_v> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("of the modulus p_v") {
    constexpr std::array<uint32_t, num_limbs_v> expect = {0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> ret;

    subtract_p(ret.data(), p_v.data());

    REQUIRE(expect == ret);
  }

  SECTION("of one above the modulus p_v") {
    constexpr std::array<uint32_t, num_limbs_v> a = {0xd87cfd48, 0x3c208c16, 0x6871ca8d,
                                                     0x97816a91, 0x8181585d, 0xb85045b6,
                                                     0xe131a029, 0x30644e72};
    constexpr std::array<uint32_t, num_limbs_v> expect = {1, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(expect == ret);
  }
}
