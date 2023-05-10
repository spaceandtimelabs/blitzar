/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/base/bit/zero_equality.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basbt;

TEST_CASE("we can determine if a region of memory is zero") {
  unsigned char bytes1[10] = {};
  REQUIRE(is_zero(bytes1, sizeof(bytes1)) == 1);

  unsigned char bytes2[10] = {0, 0, 1};
  REQUIRE(is_zero(bytes2, sizeof(bytes2)) == 0);
}
