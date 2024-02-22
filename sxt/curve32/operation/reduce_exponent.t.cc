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
#include "sxt/curve32/operation/reduce_exponent.h"

#include <array>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::c32o;

TEST_CASE("Test 1 - We can verify that reduce does not change input array A when A < "
          "p") {
  // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
  // s = p - 1 (decimal)
  // s % p = p - 1 (decimal)
  std::array<unsigned long long, 4> s = {6346243789798364140ull, 1503914060200516822ull, 0ull,
                                         1152921504606846976ull};

  SECTION("Verifying for A mod p == A holds (p = 2^252 + "
          "27742317777372353535851937790883648493)") {
    std::array<unsigned long long, 4> expected_s = s;

    reduce_exponent(reinterpret_cast<unsigned char*>(s.data()));

    REQUIRE(s == expected_s);
  }
}

TEST_CASE("Test 2 - We can verify that reduce change input array A when A >= p") {
  // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
  // s = p (decimal)
  // s % p = 0 (decimal)
  std::array<unsigned long long, 4> s = {6346243789798364141ull, 1503914060200516822ull, 0ull,
                                         1152921504606846976ull};

  SECTION("Verifying for A mod p == 0 holds (p = 2^252 + "
          "27742317777372353535851937790883648493)") {
    reduce_exponent(reinterpret_cast<unsigned char*>(s.data()));

    REQUIRE(s == std::array<unsigned long long, 4>({0, 0, 0, 0}));
  }
}

TEST_CASE("Test 3 - We can verify that reduce change input array A when A >= p") {
  // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
  // s = p + 103 (decimal)
  // s % p = 103 (decimal)
  std::array<unsigned long long, 4> s = {6346243789798364244ull, 1503914060200516822ull, 0ull,
                                         1152921504606846976ull};

  SECTION("Verifying for A mod p == 103 holds (p = 2^252 + "
          "27742317777372353535851937790883648493)") {
    reduce_exponent(reinterpret_cast<unsigned char*>(s.data()));

    REQUIRE(s == std::array<unsigned long long, 4>({103, 0, 0, 0}));
  }
}

TEST_CASE("Test 4 - We can verify that reduce correctly change input array A even "
          "when A is the maximum possible value") {
  // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
  // s =
  // 115792089237316195423570985008687907853269984665640564039457584007913129639935
  // (decimal) s % p =
  // 7237005577332262213973186563042994240413239274941949949428319933631315875100
  // (decimal)
  std::array<unsigned long long, 4> s = {18446744073709551615ull, 18446744073709551615ull,
                                         18446744073709551615ull, 18446744073709551615ull};
  std::array<unsigned long long, 4> expected_s = {15486807595281847580ull, 14334777244411350896ull,
                                                  18446744073709551614ull, 1152921504606846975ull};

  SECTION("Verifying for A mod p holds (p = 2^252 + "
          "27742317777372353535851937790883648493)") {
    reduce_exponent(reinterpret_cast<unsigned char*>(s.data()));

    REQUIRE(s == expected_s);
  }
}
