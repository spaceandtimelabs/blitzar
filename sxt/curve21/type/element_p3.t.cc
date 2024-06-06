/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/curve21/type/element_p3.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::c21t;
using sxt::f51t::operator""_f51;

TEST_CASE("todo") {
  element_p3 e{0x3b6f8891960f6ad45776d1e1213c1bd9de44f888163a76921515e6cf9f3fd67e_f51,
               0x336d9ece4cdb30925921f40f14dab827d6e156675107378db6d34c9a874a007e_f51,
               0x59e4ea1a52a20ea2fd9cb81712f675b450b27bff31b598ba722d5b0bf61c8608_f51,
               0x1f6e08da2d298daafc6ea6fedd5e07c172749500483d139bc532c7e392cad989_f51};

  SECTION("we can convert an element to compact form and back again") {
    compact_element g{static_cast<compact_element>(e)};
    element_p3 ep{g};
    REQUIRE(ep == e);
  }

  SECTION("we can convert the identity from compact form") {
    element_p3 i{compact_element::identity()};
    REQUIRE(i == element_p3::identity());
  }
}
