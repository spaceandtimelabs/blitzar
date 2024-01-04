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
#include "sxt/field12/base/reduce.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/constants.h"

using namespace sxt::f12b;

TEST_CASE("reducing using Montgomery reduction") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint64_t, 12> t = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                            0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 6> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x0);
    REQUIRE(h[1] == 0x0);
    REQUIRE(h[2] == 0x0);
    REQUIRE(h[3] == 0x0);
    REQUIRE(h[4] == 0x0);
    REQUIRE(h[5] == 0x0);
  }

  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint64_t, 12> t = {r_v[0], r_v[1], r_v[2], r_v[3], r_v[4], r_v[5],
                                            0x0,    0x0,    0x0,    0x0,    0x0,    0x0};
    std::array<uint64_t, 6> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x1);
    REQUIRE(h[1] == 0x0);
    REQUIRE(h[2] == 0x0);
    REQUIRE(h[3] == 0x0);
    REQUIRE(h[4] == 0x0);
    REQUIRE(h[5] == 0x0);
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint64_t, 12> t = {p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5],
                                            0x0,    0x0,    0x0,    0x0,    0x0,    0x0};
    std::array<uint64_t, 6> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x0);
    REQUIRE(h[1] == 0x0);
    REQUIRE(h[2] == 0x0);
    REQUIRE(h[3] == 0x0);
    REQUIRE(h[4] == 0x0);
    REQUIRE(h[5] == 0x0);
  }

  SECTION("with the modulus minus one returns pre-computed value") {
    constexpr std::array<uint64_t, 12> t = {0xb9fef13fffffaaaa,
                                            0x1eabffeb153ffff,
                                            0x673062a0f6b0f624,
                                            0x64774bd4f38512bf,
                                            0x4b1ba4b6434bacd7,
                                            0x1a0111ea397fe69a,
                                            0x0,
                                            0x0,
                                            0x0,
                                            0x0,
                                            0x0,
                                            0x0};
    std::array<uint64_t, 6> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0x807b0c0a130c339d);
    REQUIRE(h[1] == 0x877169a752e98495);
    REQUIRE(h[2] == 0xf9bcc43fcde624a6);
    REQUIRE(h[3] == 0xe3c54facb0492472);
    REQUIRE(h[4] == 0xcfefecb1742bb352);
    REQUIRE(h[5] == 0x13cd4d80bfdb157d);
  }

  SECTION("with the maximum value returns pre-computed value") {
    constexpr std::array<uint64_t, 12> t = {0xffffffffffffffff,
                                            0xffffffffffffffff,
                                            0xffffffffffffffff,
                                            0xffffffffffffffff,
                                            0xffffffffffffffff,
                                            0xffffffffffffffff,
                                            0x0,
                                            0x0,
                                            0x0,
                                            0x0,
                                            0x0,
                                            0x0};
    std::array<uint64_t, 6> h;

    reduce(h.data(), t.data());

    REQUIRE(h[0] == 0xc52b7da6c7f4628c);
    REQUIRE(h[1] == 0x9ecaed89d8bb0503);
    REQUIRE(h[2] == 0x32f22927e21b885b);
    REQUIRE(h[3] == 0x4cdfa0709adc84d6);
    REQUIRE(h[4] == 0x5dbd438f06fc594c);
    REQUIRE(h[5] == 0x5024ae85084d9b0);
  }
}

TEST_CASE("the below modulus function") {
  SECTION("returns false if above the modulus") {
    constexpr std::array<uint64_t, 6> h = {0x7cf2b39786a8ca98, 0x581d46434e7d165d,
                                           0x9e892e2caed9d420, 0xf9d008b1efaa0491,
                                           0x8b69878b1e985eeb, 0xddce78033db614b1};

    REQUIRE(is_below_modulus(h.data()) == false);
  }

  SECTION("returns true if below the modulus") {
    constexpr std::array<uint64_t, 6> h = {0xacfab39786ab7540, 0x62bd464dc3dd165f,
                                           0x65029924f95222ff, 0xd615ac8a53816e96,
                                           0x328c49d9043af830, 0xdc5e8b171b6dfdf};

    REQUIRE(is_below_modulus(h.data()) == true);
  }
}
