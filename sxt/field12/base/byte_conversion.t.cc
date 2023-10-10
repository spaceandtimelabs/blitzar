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
#include "sxt/field12/base/byte_conversion.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/constants.h"

using namespace sxt::f12b;

TEST_CASE("complete byte conversion") {
  SECTION("with pre-computed value below modulus returns original value") {
    constexpr std::array<uint8_t, 48> s = {
        26,  1,   17,  234, 57,  127, 230, 154, 75,  27,  167, 182, 67,  75,  172, 215,
        100, 119, 75,  132, 243, 133, 18,  191, 103, 48,  210, 160, 246, 176, 246, 36,
        30,  171, 255, 254, 177, 83,  255, 255, 185, 254, 255, 255, 255, 255, 170, 170};

    constexpr std::array<uint64_t, 6> expect_h = {0x43f5fffffffcaaae, 0x32b7fff2ed47fffd,
                                                  0x7e83a49a2e99d69,  0xeca8f3318332bb7a,
                                                  0xef148d1ea0f4c069, 0x40ab3263eff0206};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect_h == h);

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 48> ret;
    to_bytes(ret.data(), h.data());
    REQUIRE(ret == s);
  }

  SECTION("with a value equal to the modulus minus one returns original value") {
    constexpr std::array<uint64_t, 6> h = {0xb9feffffffffaaaa, 0x1eabfffeb153ffff,
                                           0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                           0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};

    constexpr std::array<uint8_t, 48> s = {
        5,   2,   74,  232, 80,  132, 217, 176, 93,  189, 67,  143, 6,   252, 89,  76,
        76,  223, 160, 112, 154, 220, 132, 214, 50,  242, 41,  39,  226, 27,  136, 91,
        158, 202, 237, 137, 216, 187, 5,   3,   197, 43,  125, 166, 199, 244, 98,  139};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> ret_h;
    bool is_below_modulus;
    from_bytes(is_below_modulus, ret_h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(h == ret_h);

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 48> ret_s;
    to_bytes(ret_s.data(), ret_h.data());
    REQUIRE(s == ret_s);
  }

  SECTION("with a value equal to the modulus plus one returns one") {
    constexpr std::array<uint64_t, 6> h = {0xb9feffffffffaaac, 0x1eabfffeb153ffff,
                                           0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                           0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};

    constexpr std::array<uint8_t, 48> expect_s = {
        20,  254, 199, 1,   232, 251, 12,  233, 237, 94,  100, 39,  60, 79,  83,  139,
        23,  151, 171, 20,  88,  168, 141, 233, 52,  62,  169, 121, 20, 149, 109, 200,
        127, 225, 18,  116, 216, 152, 250, 251, 244, 211, 130, 89,  56, 11,  72,  32};

    constexpr std::array<uint64_t, 6> expect_h = {1, 0, 0, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 48> s;
    to_bytes(s.data(), h.data());
    REQUIRE(expect_s == s);

    // Convert from bytes to Montgomery form.
    // Note the to_bytes function converts (p_v + 1) mod p_v
    std::array<uint64_t, 6> ret_h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, ret_h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect_h == ret_h);
  }
}

TEST_CASE("conversion from bytes") {
  SECTION("with zero returns zero") {
    constexpr std::array<uint8_t, 48> s = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    constexpr std::array<uint64_t, 6> expect = {0, 0, 0, 0, 0, 0};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(expect == h);
  }

  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 48> s = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
    REQUIRE(r_v == h);
  }

  SECTION("with a randomly generated 384-bit value below the modulus returns true") {
    // 2119871316446388761401483399807319918740238543355632831807473852398527322499749586964544683272733792986744614516032
    // 0xdc5e8b171b6dfdf328c49d9043af830d615ac8a53816e9665029924f95222ff62bd464dc3dd165facfab39786ab7540
    constexpr std::array<uint8_t, 48> s = {
        19, 211, 222, 111, 255, 41,  175, 135, 58,  159, 164, 204, 170, 87,  84,  155,
        78, 129, 248, 201, 197, 224, 108, 251, 198, 67,  102, 60,  29,  35,  24,  206,
        92, 72,  181, 2,   190, 32,  10,  167, 188, 120, 166, 166, 230, 251, 123, 133};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
  }

  SECTION("with a randomly generated 384-bit value above the modulus returns false") {
    // 34139147758219727908743802005694553171195301102867695914463938941390780526426452502506045716304859105289898794994328
    // 0xddce78033db614b18b69878b1e985eebf9d008b1efaa04919e892e2caed9d420581d46434e7d165d7cf2b39786a8ca98
    constexpr std::array<uint8_t, 48> s = {221, 206, 120, 3,   61,  182, 20,  177, 139, 105,
                                           135, 139, 30,  152, 94,  235, 158, 137, 46,  44,
                                           174, 217, 212, 32,  88,  29,  70,  67,  78,  125,
                                           22,  93,  124, 242, 179, 151, 134, 168, 202, 152};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = true;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == false);
  }
}

TEST_CASE("conversion to bytes") {
  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 48> expect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 48> ret;
    to_bytes(ret.data(), r_v.data());
    REQUIRE(expect == ret);
  }

  SECTION("with the modulus returns zero") {
    constexpr std::array<uint8_t, 48> expect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Convert from Montgomery form to bytes.
    std::array<uint8_t, 48> ret;
    to_bytes(ret.data(), p_v.data());
    REQUIRE(expect == ret);
  }
}
