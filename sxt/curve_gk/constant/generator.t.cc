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
#include "sxt/curve_gk/constant/generator.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/montgomery.h"
#include "sxt/fieldgk/type/element.h"

using namespace sxt;
using namespace sxt::cgkcn;

TEST_CASE("generator_x_v") {
  SECTION("is 1 in Montgomery form") {
    constexpr std::array<uint64_t, 4> a{1, 0, 0, 0};
    fgkt::element ret;

    fgkb::to_montgomery_form(ret.data(), a.data());

    REQUIRE(generator_x_v == ret);
  }
}

TEST_CASE("generator_y_v") {
  SECTION("is -16 in Montgomery form") {
    /**
     * Generated using SAGE:
     * p_v = 21888242871839275222246405745257275088548364400416034343698204186575808495617
     * Fq = GF(p_v)
     * hex(Fq(-16).sqrt())
     * 0x2cf135e7506a45d632d270d45f1181294833fc48d823f272c
     */
    constexpr std::array<uint64_t, 4> a{0x833fc48d823f272c, 0x2d270d45f1181294, 0xcf135e7506a45d63,
                                        0x2};

    fgkt::element ret;

    fgkb::to_montgomery_form(ret.data(), a.data());

    REQUIRE(generator_y_v == ret);
  }
}
