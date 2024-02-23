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
#include "sxt/field32/operation/sq.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("square") {
  constexpr std::array<uint8_t, 32> a = {0xa8, 0xb6, 0xe6, 0x9e, 0x46, 0x03, 0x66, 0x56,
                                         0xe8, 0x45, 0x84, 0x08, 0x97, 0x80, 0xba, 0x0d,
                                         0x15, 0x4b, 0xfa, 0xb4, 0x42, 0x8a, 0x4d, 0x84,
                                         0x58, 0x41, 0x64, 0x9c, 0xcb, 0x9f, 0x06, 0x69};
  f32t::element a_elm;
  f32b::from_bytes(a_elm.data(), a.data());

  SECTION("sq") {
    constexpr std::array<uint8_t, 32> expected = {0x19, 0xb6, 0xed, 0xa6, 0x9c, 0x64, 0x9b, 0x6a,
                                                  0x01, 0xcb, 0x10, 0xe0, 0xe2, 0x16, 0x0f, 0x59,
                                                  0xcc, 0xd6, 0xfe, 0xc7, 0x6f, 0x46, 0x6b, 0x2c,
                                                  0x71, 0x0d, 0xf6, 0x2b, 0x38, 0xbd, 0xb4, 0x6c};

    f32t::element expected_elm;
    f32b::from_bytes(expected_elm.data(), expected.data());

    f32t::element h;
    sq(h, a_elm);

    REQUIRE(h == expected_elm);
  }

  SECTION("sq2") {
    constexpr std::array<uint8_t, 32> expected = {0x45, 0x6c, 0xdb, 0x4d, 0x39, 0xc9, 0x36, 0xd5,
                                                  0x02, 0x96, 0x21, 0xc0, 0xc5, 0x2d, 0x1e, 0xb2,
                                                  0x98, 0xad, 0xfd, 0x8f, 0xdf, 0x8c, 0xd6, 0x58,
                                                  0xe2, 0x1a, 0xec, 0x57, 0x70, 0x7a, 0x69, 0x59};

    f32t::element expected_elm;
    f32b::from_bytes(expected_elm.data(), expected.data());

    f32t::element h;
    sq2(h, a_elm);

    REQUIRE(h == expected_elm);
  }
}

TEST_CASE("with literals") {
  SECTION("regular") {
    auto e = 0x48674afb484b050fdcccf508dfb8ce91c364ab4d15584711cba01736e1c59deb_f32;
    f32t::element res;
    sq(res, e);
    auto expected_res = 0x7fa13403b69cc40197d157d218f6f8afdfe95bc7e98ef46112480fe346aa6ec3_f32;
    REQUIRE(res == expected_res);
  }

  SECTION("times two") {
    auto e = 0x711a90c454965634b0962b2b4479551d887ad8d7f33d62f626648de22323dba0_f32;
    f32t::element res;
    sq2(res, e);
    auto expected_res = 0x55a6d8f01f8a9a9a2385a64a8d3aeae2c0d8895a8027b9fc8725cce6360e0f2_f32;
    REQUIRE(res == expected_res);
  }
}