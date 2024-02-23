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
#include "sxt/field32/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("neg") {
  SECTION("from bytes") {
    constexpr std::array<uint8_t, 32> a = {0xa8, 0xb6, 0xe6, 0x9e, 0x46, 0x03, 0x66, 0x56,
                                           0xe8, 0x45, 0x84, 0x08, 0x97, 0x80, 0xba, 0x0d,
                                           0x15, 0x4b, 0xfa, 0xb4, 0x42, 0x8a, 0x4d, 0x84,
                                           0x58, 0x41, 0x64, 0x9c, 0xcb, 0x9f, 0x06, 0x69};
    f32t::element a_elm;
    f32b::from_bytes(a_elm.data(), a.data());

    constexpr std::array<uint8_t, 32> expected = {0x45, 0x49, 0x19, 0x61, 0xb9, 0xfc, 0x99, 0xa9,
                                                  0x17, 0xba, 0x7b, 0xf7, 0x68, 0x7f, 0x45, 0xf2,
                                                  0xea, 0xb4, 0x05, 0x4b, 0xbd, 0x75, 0xb2, 0x7b,
                                                  0xa7, 0xbe, 0x9b, 0x63, 0x34, 0x60, 0xf9, 0x16};
    f32t::element expected_elm;
    f32b::from_bytes(expected_elm.data(), expected.data());

    f32t::element h;
    neg(h, a_elm);

    REQUIRE(h == expected_elm);
  }

  SECTION("from literal") {
    auto a = 0x5187889dbf41151c5858a2cc5924e0a9e34592749088b9866d452769294eca83_f32;
    f32t::element ret;
    neg(ret, a);
    REQUIRE(ret == 0x2e78776240beeae3a7a75d33a6db1f561cba6d8b6f77467992bad896d6b1356a_f32);
  }
}
