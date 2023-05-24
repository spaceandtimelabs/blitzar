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
#include "sxt/scalar25/type/literal.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::s25t;

TEST_CASE("literals are valid") {
  REQUIRE(element{0u} == 0x00_s25);
  REQUIRE(element{3u} == 0x3_s25);

  // 2^252 + 27742317777372353535851937790883648493
  element e;
  std::array<uint64_t, 4> s = {6346243789798364141ull, 1503914060200516822ull, 0ull,
                               1152921504606846976ull};
  memcpy(e.data(), s.data(), 32);
  REQUIRE(e == 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25);
}
