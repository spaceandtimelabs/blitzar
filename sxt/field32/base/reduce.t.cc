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
#include "sxt/field32/base/reduce.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/constants.h"

using namespace sxt::f32b;

TEST_CASE("reducing using Montgomery reduction") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint32_t, 2 * num_limbs_v> t = {0};
    std::array<uint32_t, num_limbs_v> h;

    reduce(h.data(), t.data());

    for (unsigned i = 0; i < num_limbs_v; ++i) {
      REQUIRE(h[i] == 0);
    }
  }

  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint32_t, 2 * num_limbs_v> t = {
        r_v[0], r_v[1], r_v[2], r_v[3], r_v[4], r_v[5], r_v[6], r_v[7], 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x1);
    for (unsigned i = 1; i < num_limbs_v; ++i) {
      REQUIRE(h[i] == 0);
    }
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint32_t, 2 * num_limbs_v> t = {
        p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5], p_v[6], p_v[7], 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, num_limbs_v> h;

    reduce(h.data(), t.data());

    for (unsigned i = 0; i < num_limbs_v; ++i) {
      REQUIRE(h[i] == 0);
    }
  }

  SECTION("with r2_v returns r_v") {
    constexpr std::array<uint32_t, 2 * num_limbs_v> t = {
        r2_v[0], r2_v[1], r2_v[2], r2_v[3], r2_v[4], r2_v[5], r2_v[6], r2_v[7],
        0,       0,       0,       0,       0,       0,       0,       0};

    std::array<uint32_t, num_limbs_v> h;

    reduce(h.data(), t.data());

    for (unsigned i = 0; i < num_limbs_v; ++i) {
      REQUIRE(h[i] == r_v[i]);
    }
  }
}

TEST_CASE("the below modulus function") {
  SECTION("returns true if one below the modulus p_v") {
    constexpr std::array<uint32_t, num_limbs_v> h = {0xd87cfd46, 0x3c208c16, 0x6871ca8d,
                                                     0x97816a91, 0x8181585d, 0xb85045b6,
                                                     0xe131a029, 0x30644e72};

    REQUIRE(is_below_modulus(h.data()) == true);
  }

  SECTION("returns false if equal to the modulus p_v") {
    REQUIRE(is_below_modulus(p_v.data()) == false);
  }

  SECTION("returns false if one above the modulus p_v") {
    constexpr std::array<uint32_t, num_limbs_v> h = {0xd87cfd48, 0x3c208c16, 0x6871ca8d,
                                                     0x97816a91, 0x8181585d, 0xb85045b6,
                                                     0xe131a029, 0x30644e72};

    REQUIRE(is_below_modulus(h.data()) == false);
  }
}
