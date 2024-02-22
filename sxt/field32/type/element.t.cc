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
#include "sxt/field32/type/element.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"

using namespace sxt::f32t;

TEST_CASE("element conversion") {
  std::ostringstream oss;
  SECTION("of zero prints as zero") {
    element e{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    oss << e;
    REQUIRE(oss.str() == "0x0_f32");
  }

  SECTION("of one prints as one") {
    element e{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    oss << e;
    REQUIRE(oss.str() == "0x1_f32");
  }

  SECTION("of Edwards D prints as expected") {
    element e{56195235, 13857412, 51736253, 6949390, 114729,
              24766616, 60832955, 30306712, 48412415, 21499315};
    oss << e;
    REQUIRE(oss.str() == "0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3_f32");
  }
}

TEST_CASE("element equality") {
  element e{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  REQUIRE(e == e);
}
