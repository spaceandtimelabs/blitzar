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
#include "sxt/fieldgk/base/reduce.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/constants.h"

using namespace sxt::fgkb;

TEST_CASE("reducing using Montgomery reduction") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint64_t, 8> t = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 4> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x0);
    REQUIRE(h[1] == 0x0);
    REQUIRE(h[2] == 0x0);
    REQUIRE(h[3] == 0x0);
  }

  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint64_t, 8> t = {r_v[0], r_v[1], r_v[2], r_v[3], 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 4> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x1);
    REQUIRE(h[1] == 0x0);
    REQUIRE(h[2] == 0x0);
    REQUIRE(h[3] == 0x0);
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint64_t, 8> t = {p_v[0], p_v[1], p_v[2], p_v[3], 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 4> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x0);
    REQUIRE(h[1] == 0x0);
    REQUIRE(h[2] == 0x0);
    REQUIRE(h[3] == 0x0);
  }

  SECTION("with r2_v returns r_v") {
    constexpr std::array<uint64_t, 8> t = {r2_v[0], r2_v[1], r2_v[2], r2_v[3], 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 4> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == r_v[0]);
    REQUIRE(h[1] == r_v[1]);
    REQUIRE(h[2] == r_v[2]);
    REQUIRE(h[3] == r_v[3]);
  }
}

TEST_CASE("the below modulus function") {
  SECTION("returns true if one below the modulus p_v") {
    constexpr std::array<uint64_t, 4> h = {0x43e1f593f0000000, 0x2833e84879b97091,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    REQUIRE(is_below_modulus(h.data()) == true);
  }

  SECTION("returns false if equal to the modulus p_v") {
    REQUIRE(is_below_modulus(p_v.data()) == false);
  }

  SECTION("returns false if one above the modulus p_v") {
    constexpr std::array<uint64_t, 4> h = {0x43e1f593f0000002, 0x2833e84879b97091,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    REQUIRE(is_below_modulus(h.data()) == false);
  }
}
