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
#include "sxt/ristretto/operation/add.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;
using namespace sxt::rsto;

TEST_CASE("we can add curve elements") {
  rstt::compressed_element g{226, 242, 174, 10,  106, 188, 78,  113, 168, 132, 169,
                             97,  197, 0,   81,  95,  88,  227, 11,  106, 165, 130,
                             221, 141, 182, 166, 89,  69,  224, 141, 45,  118};

  rstt::compressed_element res;

  SECTION("adding zero acts as the identity function") {
    add(res, rstt::compressed_element{}, g);
    REQUIRE(res == g);
  }
}
