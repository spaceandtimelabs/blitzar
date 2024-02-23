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
#include "sxt/field32/operation/sub.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("sub") {
  SECTION("with bytes") {

    constexpr std::array<uint8_t, 32> a = {0xa8, 0xb6, 0xe6, 0x9e, 0x46, 0x03, 0x66, 0x56,
                                           0xe8, 0x45, 0x84, 0x08, 0x97, 0x80, 0xba, 0x0d,
                                           0x15, 0x4b, 0xfa, 0xb4, 0x42, 0x8a, 0x4d, 0x84,
                                           0x58, 0x41, 0x64, 0x9c, 0xcb, 0x9f, 0x06, 0x69};
    f32t::element a_elm;
    f32b::from_bytes(a_elm.data(), a.data());

    constexpr std::array<uint8_t, 32> b = {0x6f, 0xda, 0x36, 0x6f, 0x26, 0x9f, 0xc7, 0x54,
                                           0x9d, 0xfe, 0xf8, 0xcc, 0x9d, 0xd8, 0x96, 0x4a,
                                           0x8e, 0xb0, 0x81, 0xaa, 0x50, 0x03, 0x90, 0x9a,
                                           0x15, 0x32, 0xd0, 0x72, 0x48, 0x5f, 0xe5, 0x76};
    f32t::element b_elm;
    f32b::from_bytes(b_elm.data(), b.data());

    constexpr std::array<uint8_t, 32> expected = {0x26, 0xdc, 0xaf, 0x2f, 0x20, 0x64, 0x9e, 0x01,
                                                  0x4b, 0x47, 0x8b, 0x3b, 0xf9, 0xa7, 0x23, 0xc3,
                                                  0x86, 0x9a, 0x78, 0x0a, 0xf2, 0x86, 0xbd, 0xe9,
                                                  0x42, 0x0f, 0x94, 0x29, 0x83, 0x40, 0x21, 0x72};
    f32t::element expected_elm;
    f32b::from_bytes(expected_elm.data(), expected.data());

    f32t::element h;
    sub(h, a_elm, b_elm);

    REQUIRE(h == expected_elm);
  }

  SECTION("with literals") {
    auto a = 0x5187889dbf41151c5858a2cc5924e0a9e34592749088b9866d452769294eca83_f32;
    auto b = 0x7397e3032f73c74bb068e931363dc16ea16bb74d68e16d331ca2f087c70eef70_f32;
    auto expected = 0x5defa59a8fcd4dd0a7efb99b22e71f3b41d9db2727a74c5350a236e1623fdb00_f32;

    f32t::element diff;

    sub(diff, a, b);

    REQUIRE(diff == expected);
  }
}
