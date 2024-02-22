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
#include "sxt/field32/base/byte_conversion.h"

#include <array>

#include "sxt/base/test/unit_test.h"

using namespace sxt::f32b;

TEST_CASE("byte_conversion") {
  SECTION("from_bytes works") {
    constexpr std::array<uint8_t, 32> bytes = {0x38, 0x4e, 0xc9, 0xf4, 0x66, 0x76, 0x02, 0xd6,
                                               0xf7, 0xb8, 0xb3, 0x3c, 0xb5, 0x83, 0xf2, 0x0d,
                                               0x1c, 0x91, 0x6d, 0xaa, 0x8d, 0xf0, 0x62, 0xc4,
                                               0x5c, 0x86, 0xf2, 0xf4, 0x91, 0x61, 0xb8, 0x03};
    constexpr std::array<uint32_t, 10> expected = {13192760, 10328509, 52361920, 27911581, 3656206,
                                                   7180572,  24659669, 13342860, 18829096, 975238};
    uint32_t h[10];

    from_bytes(h, bytes.data());

    for (int i = 0; i < 10; i++) {
      REQUIRE(h[i] == expected[i]);
    }
  }

  SECTION("to_bytes works") {
    constexpr std::array<uint32_t, 10> h = {13192760, 10328509, 52361920, 27911581, 3656206,
                                            7180572,  24659669, 13342860, 18829096, 975238};
    constexpr std::array<uint8_t, 32> expected = {0x38, 0x4e, 0xc9, 0xf4, 0x66, 0x76, 0x02, 0xd6,
                                                  0xf7, 0xb8, 0xb3, 0x3c, 0xb5, 0x83, 0xf2, 0x0d,
                                                  0x1c, 0x91, 0x6d, 0xaa, 0x8d, 0xf0, 0x62, 0xc4,
                                                  0x5c, 0x86, 0xf2, 0xf4, 0x91, 0x61, 0xb8, 0x03};

    std::array<uint8_t, 32> ret;

    to_bytes(ret.data(), h.data());

    for (int i = 0; i < 32; i++) {
      REQUIRE(ret[i] == expected[i]);
    }
  }
}
