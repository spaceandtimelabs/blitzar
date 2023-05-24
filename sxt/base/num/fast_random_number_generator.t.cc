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
#include "sxt/base/num/fast_random_number_generator.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basn;

TEST_CASE("fast_random_number_generator can be used to quickly generate a sequence "
          "of random numbers") {
  fast_random_number_generator generator1{1, 2};
  auto x1 = generator1();
  auto x2 = generator1();
  REQUIRE(x1 != x2);
}
