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
#include "sxt/curve_gkg1/constant/b.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/montgomery.h"
#include "sxt/fieldgk/type/element.h"

using namespace sxt;
using namespace sxt::ck1cn;

TEST_CASE("b_v") {
  SECTION("is 3 in Montgomery form") {
    constexpr std::array<uint64_t, 4> a{3, 0, 0, 0};
    fgkt::element ret;

    fgkb::to_montgomery_form(ret.data(), a.data());

    REQUIRE(b_v == ret);
  }
}
