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
#include "sxt/fieldgk/base/byte_conversion.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/constants.h"

using namespace sxt::fgkb;

TEST_CASE("complete byte conversion in big endian") {
  SECTION("with a value equal to the modulus minus one returns original value") {
    constexpr std::array<uint64_t, 4> h = {0x43e1f593f0000000, 0x2833e84879b97091,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> s = {0x1a, 0x78, 0x55, 0x21, 0x5e, 0x6c, 0x4b, 0x0c,
                                           0xf0, 0x2a, 0x37, 0xd1, 0xd2, 0xc8, 0xfb, 0x00,
                                           0x1f, 0x24, 0xf2, 0x9e, 0x98, 0xa7, 0x84, 0x09,
                                           0x67, 0x86, 0x55, 0x8e, 0x82, 0x4e, 0xe6, 0xb3};

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
    constexpr std::array<uint64_t, 4> h = {0x43e1f593f0000002, 0x2833e84879b97091,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> expect_s = {0x15, 0xeb, 0xf9, 0x51, 0x82, 0xc5, 0x55, 0x1c,
                                                  0xc8, 0x26, 0x0d, 0xe4, 0xae, 0xb8, 0x5d, 0x5d,
                                                  0x09, 0x0e, 0xf5, 0xa9, 0xe1, 0x11, 0xec, 0x87,
                                                  0xdc, 0x5b, 0xa0, 0x05, 0x6d, 0xb1, 0x19, 0x4e};

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
    constexpr std::array<uint64_t, 4> h = {0x43e1f593f0000000, 0x2833e84879b97091,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> s = {0xb3, 0xe6, 0x4e, 0x82, 0x8e, 0x55, 0x86, 0x67,
                                           0x09, 0x84, 0xa7, 0x98, 0x9e, 0xf2, 0x24, 0x1f,
                                           0x00, 0xfb, 0xc8, 0xd2, 0xd1, 0x37, 0x2a, 0xf0,
                                           0x0c, 0x4b, 0x6c, 0x5e, 0x21, 0x55, 0x78, 0x1a};

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
    constexpr std::array<uint64_t, 4> h = {0x43e1f593f0000002, 0x2833e84879b97091,
                                           0xb85045b68181585d, 0x30644e72e131a029};

    constexpr std::array<uint8_t, 32> expect_s = {0x4e, 0x19, 0xb1, 0x6d, 0x05, 0xa0, 0x5b, 0xdc,
                                                  0x87, 0xec, 0x11, 0xe1, 0xa9, 0xf5, 0x0e, 0x09,
                                                  0x5d, 0x5d, 0xb8, 0xae, 0xe4, 0x0d, 0x26, 0xc8,
                                                  0x1c, 0x55, 0xc5, 0x82, 0x51, 0xf9, 0xeb, 0x15};

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
