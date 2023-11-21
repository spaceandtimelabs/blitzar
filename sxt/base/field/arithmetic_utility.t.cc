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
#include "sxt/base/field/arithmetic_utility.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basf;

TEST_CASE("mac (multiplication and carry) can handle computation") {
  SECTION("with minimum values") {
    constexpr uint64_t a{0x0};
    constexpr uint64_t b{0x0};
    constexpr uint64_t c{0x0};
    uint64_t carry{0x0};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x0);
  }

  SECTION("without carryover on pre-comuputed values") {
    constexpr uint64_t a{0x1};
    constexpr uint64_t b{0x2};
    constexpr uint64_t c{0x3};
    uint64_t carry{0x4};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xb);
    REQUIRE(carry == 0x0);
  }

  SECTION("with carryover on pre-comuputed values") {
    constexpr uint64_t a{0xb9feffffffffaaab};
    constexpr uint64_t b{0x1eabfffeb153ffff};
    constexpr uint64_t c{0x6730d2a0f6b0f624};
    uint64_t carry{0x64774b84f38512bf};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xc974d8dba4a3c746);
    REQUIRE(carry == 0xc5d0d7bda277422);
  }

  SECTION("with maximum values") {
    constexpr uint64_t a{0xffffffffffffffff};
    constexpr uint64_t b{0xffffffffffffffff};
    constexpr uint64_t c{0xffffffffffffffff};
    uint64_t carry{0xffffffffffffffff};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xffffffffffffffff);
    REQUIRE(carry == 0xffffffffffffffff);
  }
}

TEST_CASE("adc (addition and carry) can handle computation") {
  SECTION("with minimum values") {
    constexpr uint64_t a{0x0};
    constexpr uint64_t b{0x0};
    uint64_t carry{0x0};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x0);
  }

  SECTION("without carryover on pre-comuputed values") {
    constexpr uint64_t a{0x3};
    constexpr uint64_t b{0x4};
    uint64_t carry{0xa};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x11);
    REQUIRE(carry == 0x0);
  }

  SECTION("with carryover on pre-comuputed values") {
    constexpr uint64_t a{0x1};
    constexpr uint64_t b{0xffffffffffffffff};
    uint64_t carry{0x0};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x1);
  }

  SECTION("with maximum values") {
    constexpr uint64_t a{0xffffffffffffffff};
    constexpr uint64_t b{0xffffffffffffffff};
    uint64_t carry{0xffffffffffffffff};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0xfffffffffffffffd);
    REQUIRE(carry == 0x2);
  }
}

TEST_CASE("sbb (subtraction and borrow) can handle computation") {
  SECTION("with left summand less than than right summand and no borrow") {
    constexpr uint64_t a{0x031ff1ffffffffff};
    constexpr uint64_t b{0xeffff3ff5ff2feff};
    uint64_t borrow{0};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0x131ffe00a00d0100);
    REQUIRE(borrow == 0xffffffffffffffff);
  }

  SECTION("with left summand less than than right summand and borrow") {
    constexpr uint64_t a{0x031ff1f7f13ff4f2};
    constexpr uint64_t b{0xe078f3ff5ff2fef9};
    uint64_t borrow{0xdd5902076eb30a06};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0x22a6fdf8914cf5f8);
    REQUIRE(borrow == 0xffffffffffffffff);
  }

  SECTION("with left summand greater than right summand and no borrow") {
    constexpr uint64_t a{0xeffff3ff5ff2feff};
    constexpr uint64_t b{0x031ff1ffffffffff};
    uint64_t borrow{0};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0xece001ff5ff2ff00);
    REQUIRE(borrow == 0);
  }

  SECTION("with left summand greater than right summand and borrow") {
    constexpr uint64_t a{0xe078f3ff5ff2fef9};
    constexpr uint64_t b{0x031ff1f7f13ff4f2};
    uint64_t borrow{0xdd5902076eb30a06};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0xdd5902076eb30a06);
    REQUIRE(borrow == 0x0);
  }
}
