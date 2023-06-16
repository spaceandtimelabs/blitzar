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
#include "sxt/field12/base/arithmetic_utility.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/constants.h"

using namespace sxt::f12b;

TEST_CASE("we can do multiplication and carry (mac) as expected") {
  SECTION("all zeros return all zeros") {
    constexpr uint64_t a = 0x0;
    constexpr uint64_t b = 0x0;
    constexpr uint64_t c = 0x0;
    uint64_t carry = 0x0;
    uint64_t ret = 0x0;
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x0);
  }

  SECTION("small pre computed values return expected values") {
    constexpr uint64_t a = 0x1;
    constexpr uint64_t b = 0x2;
    constexpr uint64_t c = 0x3;
    uint64_t carry = 0x4;
    uint64_t ret = 0x0;
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xb);
    REQUIRE(carry == 0x0);
  }

  SECTION("large pre computed values return expected values") {
    constexpr uint64_t a = 0xb9feffffffffaaab;
    constexpr uint64_t b = 0x1eabfffeb153ffff;
    constexpr uint64_t c = 0x6730d2a0f6b0f624;
    uint64_t carry = 0x64774b84f38512bf;
    uint64_t ret = 0x0;
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xc974d8dba4a3c746);
    REQUIRE(carry == 0xc5d0d7bda277422);
  }

  SECTION("the largest possible values return expected values") {
    constexpr uint64_t a = 0xffffffffffffffff;
    constexpr uint64_t b = 0xffffffffffffffff;
    constexpr uint64_t c = 0xffffffffffffffff;
    uint64_t carry = 0xffffffffffffffff;
    uint64_t ret = 0x0;
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xffffffffffffffff);
    REQUIRE(carry == 0xffffffffffffffff);
  }
}

TEST_CASE("we can do addition and carry (adc) as expected") {
  SECTION("small pre computed values return expected values") {
    constexpr uint64_t a = 0x3;
    constexpr uint64_t b = 0x4;
    uint64_t carry = 0xa;
    uint64_t ret = 0x0;
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x11);
    REQUIRE(carry == 0x0);
  }

  SECTION("operation with small carry over return expected values") {
    constexpr uint64_t a = 0x1;
    constexpr uint64_t b = 0xffffffffffffffff;
    uint64_t carry = 0x0;
    uint64_t ret = 0x0;
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x1);
  }

  SECTION("the largest possible values return expected values") {
    constexpr uint64_t a = 0xffffffffffffffff;
    constexpr uint64_t b = 0xffffffffffffffff;
    uint64_t carry = 0xffffffffffffffff;
    uint64_t ret = 0x0;
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0xfffffffffffffffd);
    REQUIRE(carry == 0x2);
  }
}

TEST_CASE("we can do subtraction and borrow (sbb) as expected") {
  SECTION("pre computed values, a < b, and zero borrow return expected values") {
    constexpr uint64_t a = 0x031ff1ffffffffff;
    constexpr uint64_t b = 0xeffff3ff5ff2feff;
    uint64_t borrow = 0;
    uint64_t ret;
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 1378099289637323008);
    REQUIRE(borrow == 18446744073709551615);
  }

  SECTION("pre computed values, a > b, and zero borrow return expected values") {
    constexpr uint64_t a = 0xeffff3ff5ff2feff;
    constexpr uint64_t b = 0x031ff1ffffffffff;
    uint64_t borrow = 0;
    uint64_t ret;
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 17068644784072228608);
    REQUIRE(borrow == 0);
  }

  SECTION("pre computed values, a < b, and non-zero borrow return expected values") {
    constexpr uint64_t a = 0x031ff1f7f13ff4f2;
    constexpr uint64_t b = 0xe078f3ff5ff2fef9;
    uint64_t borrow = 0xdd5902076eb30a06;
    uint64_t ret;
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 2496962287454975480);
    REQUIRE(borrow == 18446744073709551615);
  }

  SECTION("pre computed values, a > b, and non-zero borrow return expected values") {
    constexpr uint64_t a = 0xe078f3ff5ff2fef9;
    constexpr uint64_t b = 0x031ff1f7f13ff4f2;
    uint64_t borrow = 0xdd5902076eb30a06;
    uint64_t ret;
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 15949781786254576134);
    REQUIRE(borrow == 0);
  }
}

TEST_CASE("we can do subtraction the modulous (subtract_p) as expected") {
  SECTION("all zeros return all zeros") {
    constexpr std::array<uint64_t, 6> a = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    constexpr std::array<uint64_t, 6> p = {p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5]};
    std::array<uint64_t, 6> ret;
    constexpr std::array<uint64_t, 6> expect = a;

    subtract_p(ret.data(), a.data(), p.data());

    REQUIRE(expect == ret);
  }

  SECTION("value is below the modulus returns the same value") {
    constexpr std::array<uint64_t, 6> a = {0xb9feffffffffaaaa, 0x1eabfffeb153ffff,
                                           0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                           0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr std::array<uint64_t, 6> p = {p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5]};
    std::array<uint64_t, 6> ret;
    constexpr std::array<uint64_t, 6> expect = {a[0], a[1], a[2], a[3], a[4], a[5]};

    subtract_p(ret.data(), a.data(), p.data());

    REQUIRE(expect == ret);
  }

  SECTION("value equal to the modulus returns zeros") {
    constexpr std::array<uint64_t, 6> p = {p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5]};
    std::array<uint64_t, 6> ret;
    constexpr std::array<uint64_t, 6> expect = {0, 0, 0, 0, 0, 0};

    subtract_p(ret.data(), p.data(), p.data());

    REQUIRE(expect == ret);
  }

  SECTION("value above the modulus returns as expected") {
    constexpr std::array<uint64_t, 6> a = {0xb9feffffffffaaae, 0x1eabfffeb153ffff,
                                           0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                           0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr std::array<uint64_t, 6> p = {p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5]};
    std::array<uint64_t, 6> ret;
    constexpr std::array<uint64_t, 6> expect = {0x3, 0x0, 0x0, 0x0, 0x0, 0x0};

    subtract_p(ret.data(), a.data(), p.data());

    REQUIRE(expect == ret);
  }

  SECTION("value at the maximum value returns as expected") {
    constexpr std::array<uint64_t, 6> a = {0xffffffffffffffff, 0xffffffffffffffff,
                                           0xffffffffffffffff, 0xffffffffffffffff,
                                           0xffffffffffffffff, 0xffffffffffffffff};
    constexpr std::array<uint64_t, 6> b = {p_v[0], p_v[1], p_v[2], p_v[3], p_v[4], p_v[5]};
    std::array<uint64_t, 6> ret;
    constexpr std::array<uint64_t, 6> expect = {0x4601000000005554, 0xe15400014eac0000,
                                                0x98cf2d5f094f09db, 0x9b88b47b0c7aed40,
                                                0xb4e45849bcb45328, 0xe5feee15c6801965};

    subtract_p(ret.data(), a.data(), b.data());

    REQUIRE(expect == ret);
  }
}
