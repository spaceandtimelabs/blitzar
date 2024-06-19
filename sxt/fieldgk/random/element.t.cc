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
#include "sxt/fieldgk/random/element.h"

#include <cstdint>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/type/element.h"

using namespace sxt;
using namespace sxt::fgkrn;

TEST_CASE("random element generation") {
  basn::fast_random_number_generator rng{1, 2};

  SECTION("will return different values if called multiple times") {
    fgkt::element e1, e2;
    generate_random_element(e1, rng);
    generate_random_element(e2, rng);
    REQUIRE(e1 != e2);
  }
}
