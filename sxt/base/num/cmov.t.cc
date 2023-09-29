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
#include "sxt/base/num/cmov.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basn;

TEST_CASE("cmov correctly moves") {
  SECTION("uint8_t elements") {
    uint8_t h{0};
    uint8_t g{0};
    constexpr uint8_t i{1};

    cmov(h, i, true);
    cmov(g, i, false);

    REQUIRE(h == i);
    REQUIRE(g == 0);
  }

  SECTION("uint16_t elements") {
    uint16_t h{0};
    uint16_t g{0};
    constexpr uint16_t i{1};

    cmov(h, i, true);
    cmov(g, i, false);

    REQUIRE(h == i);
    REQUIRE(g == 0);
  }

  SECTION("uint32_t elements") {
    uint32_t h{0};
    uint32_t g{0};
    constexpr uint32_t i{1};

    cmov(h, i, true);
    cmov(g, i, false);

    REQUIRE(h == i);
    REQUIRE(g == 0);
  }

  SECTION("uint64_t elements") {
    uint64_t h{0};
    uint64_t g{0};
    constexpr uint64_t i{1};

    cmov(h, i, true);
    cmov(g, i, false);

    REQUIRE(h == 1);
    REQUIRE(g == 0);
  }

  SECTION("signed int elements") {
    int h{0};
    int g{0};
    constexpr int i{-1};

    cmov(h, i, true);
    cmov(g, i, false);

    REQUIRE(h == i);
    REQUIRE(g == 0);
  }
}
