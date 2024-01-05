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
#include "sxt/field_bnq/base/byte_conversion.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field_bnq/base/constants.h"

using namespace sxt::fbnqb;

TEST_CASE("complete byte conversion in big endian") {
  SECTION("with a value equal to the modulus minus one returns original value") {
    constexpr std::array<uint64_t, 4> h = {0x3c208c16d87cfd46, 0x97816a916871ca8d,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> s = {0x1,  0xfd, 0x39, 0x1,  0x87, 0x4b, 0xd9, 0xef,
                                           0xe8, 0xec, 0x5b, 0xe6, 0xca, 0x3c, 0xc5, 0x83,
                                           0xac, 0x61, 0x48, 0xc,  0x65, 0xf8, 0xdc, 0x94,
                                           0x4e, 0x9c, 0x3,  0xcc, 0xd7, 0x32, 0x3,  0x10};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> ret_h;
    bool is_below_modulus;
    from_bytes(is_below_modulus, ret_h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(h == ret_h);

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> ret_s;
    to_bytes(ret_s.data(), ret_h.data());
    REQUIRE(s == ret_s);
  }

  SECTION("with a value equal to the modulus plus one returns one") {
    constexpr std::array<uint64_t, 4> h = {0x3c208c16d87cfd48, 0x97816a916871ca8d,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> expect_s = {0x2e, 0x67, 0x15, 0x71, 0x59, 0xe5, 0xc6, 0x39,
                                                  0xcf, 0x63, 0xe9, 0xcf, 0xb7, 0x44, 0x92, 0xd9,
                                                  0xeb, 0x20, 0x22, 0x85, 0x2,  0x78, 0xed, 0xf8,
                                                  0xed, 0x84, 0x88, 0x4a, 0x1,  0x4a, 0xfa, 0x37};

    constexpr std::array<uint64_t, 4> expect_h = {1, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> s;
    to_bytes(s.data(), h.data());
    REQUIRE(expect_s == s);

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> ret_h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, ret_h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect_h == ret_h);
  }
}

TEST_CASE("complete byte conversion in little endian") {
  SECTION("with a value equal to the modulus minus one returns original value") {
    constexpr std::array<uint64_t, 4> h = {0x3c208c16d87cfd46, 0x97816a916871ca8d,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> s = {0x10, 0x3,  0x32, 0xd7, 0xcc, 0x3,  0x9c, 0x4e,
                                           0x94, 0xdc, 0xf8, 0x65, 0xc,  0x48, 0x61, 0xac,
                                           0x83, 0xc5, 0x3c, 0xca, 0xe6, 0x5b, 0xec, 0xe8,
                                           0xef, 0xd9, 0x4b, 0x87, 0x1,  0x39, 0xfd, 0x1};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> ret_h;
    bool is_below_modulus;
    from_bytes_le(is_below_modulus, ret_h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(h == ret_h);

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> ret_s;
    to_bytes_le(ret_s.data(), ret_h.data());
    REQUIRE(s == ret_s);
  }

  SECTION("with a value equal to the modulus plus one returns one") {
    constexpr std::array<uint64_t, 4> h = {0x3c208c16d87cfd48, 0x97816a916871ca8d,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> expect_s = {0x37, 0xfa, 0x4a, 0x1,  0x4a, 0x88, 0x84, 0xed,
                                                  0xf8, 0xed, 0x78, 0x2,  0x85, 0x22, 0x20, 0xeb,
                                                  0xd9, 0x92, 0x44, 0xb7, 0xcf, 0xe9, 0x63, 0xcf,
                                                  0x39, 0xc6, 0xe5, 0x59, 0x71, 0x15, 0x67, 0x2e};

    constexpr std::array<uint64_t, 4> expect_h = {1, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> s;
    to_bytes_le(s.data(), h.data());
    REQUIRE(expect_s == s);

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> ret_h;
    bool is_below_modulus = false;
    from_bytes_le(is_below_modulus, ret_h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect_h == ret_h);
  }
}

TEST_CASE("conversion from big endian bytes") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint8_t, 32> s = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    constexpr std::array<uint64_t, 4> expect = {0, 0, 0, 0};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect == h);
  }

  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 32> s = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(r_v == h);
  }

  SECTION("with a value one above the modulus returns false") {
    constexpr std::array<uint8_t, 32> s = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                           255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                           255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> h;
    bool is_below_modulus = true;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == false);
  }
}

TEST_CASE("conversion from little endian bytes") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint8_t, 32> s = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    constexpr std::array<uint64_t, 4> expect = {0, 0, 0, 0};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> h;
    bool is_below_modulus = false;
    from_bytes_le(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect == h);
  }

  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 32> s = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> h;
    bool is_below_modulus = false;
    from_bytes_le(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(r_v == h);
  }

  SECTION("with a value one above the modulus returns false") {
    constexpr std::array<uint8_t, 32> s = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                           255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                           255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 4> h;
    bool is_below_modulus = true;
    from_bytes_le(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == false);
  }
}

TEST_CASE("conversion to big endian bytes") {
  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 32> expect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> ret;
    to_bytes(ret.data(), r_v.data());
    REQUIRE(expect == ret);
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint8_t, 32> expect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> ret;
    to_bytes(ret.data(), p_v.data());
    REQUIRE(expect == ret);
  }
}

TEST_CASE("conversion to little endian bytes") {
  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 32> expect = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> ret;
    to_bytes_le(ret.data(), r_v.data());
    REQUIRE(expect == ret);
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint8_t, 32> expect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 32> ret;
    to_bytes_le(ret.data(), p_v.data());
    REQUIRE(expect == ret);
  }
}
