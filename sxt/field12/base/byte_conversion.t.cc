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
        170, 170, 255, 255, 255, 255, 254, 185, 255, 255, 83,  177, 254, 255, 171, 30,
        36,  246, 176, 246, 160, 210, 48,  103, 191, 18,  133, 243, 132, 75,  119, 100,
        215, 172, 75,  67,  182, 167, 27,  75,  154, 230, 127, 57,  234, 17,  1,   26};

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
        139, 98,  244, 199, 166, 125, 43,  197, 3,   5,   187, 216, 137, 237, 202, 158,
        91,  136, 27,  226, 39,  41,  242, 50,  214, 132, 220, 154, 112, 160, 223, 76,
        76,  89,  252, 6,   143, 67,  189, 93,  176, 217, 132, 80,  232, 74,  2,   5};

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
        32,  72,  11,  56, 89,  130, 211, 244, 251, 250, 152, 216, 116, 18,  225, 127,
        200, 109, 149, 20, 121, 169, 62,  52,  233, 141, 168, 88,  20,  171, 151, 23,
        139, 83,  79,  60, 39,  100, 94,  237, 233, 12,  251, 232, 1,   199, 254, 20};

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
    constexpr std::array<uint8_t, 48> s = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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
        133, 123, 251, 230, 166, 166, 120, 188, 167, 10,  32,  190, 2,   181, 72,  92,
        102, 206, 24,  35,  29,  60,  67,  198, 251, 108, 224, 197, 201, 248, 129, 78,
        155, 84,  87,  170, 204, 164, 159, 58,  135, 175, 41,  255, 111, 222, 211, 19};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = false;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == true);
  }

  SECTION("with a randomly generated 384-bit value above the modulus returns false") {
    // 34139147758219727908743802005694553171195301102867695914463938941390780526426452502506045716304859105289898794994328
    // 0xddce78033db614b18b69878b1e985eebf9d008b1efaa04919e892e2caed9d420581d46434e7d165d7cf2b39786a8ca98
    constexpr std::array<uint8_t, 48> s = {
        152, 202, 168, 134, 151, 179, 242, 124, 93,  22, 125, 78,  67,  70,  29,  88,
        32,  212, 217, 174, 44,  46,  137, 158, 145, 4,  170, 239, 177, 8,   208, 249,
        235, 94,  152, 30,  139, 135, 105, 139, 177, 20, 182, 61,  3,   120, 206, 221};

    // Convert from bytes to Montgomery form.
    std::array<uint64_t, 6> h;
    bool is_below_modulus = true;
    from_bytes(is_below_modulus, h.data(), s.data());
    REQUIRE(is_below_modulus == false);
  }
}

TEST_CASE("conversion to bytes") {
  SECTION("with one in Montgomery form returns one") {
    constexpr std::array<uint8_t, 48> expect = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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
