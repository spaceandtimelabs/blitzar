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
#include "sxt/cbindings/backend/computational_backend_utility.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::cbnbck;

TEST_CASE("we can make a span for the referenced scalars") {
  uint8_t data[16];

  std::vector<unsigned> output_bit_table, output_lengths;

  SECTION("we handle an output of length 1") {
    output_bit_table = {1};
    output_lengths = {1};
    auto span = make_scalars_span(data, output_bit_table, output_lengths);
    REQUIRE(span.size() == 1);
    REQUIRE(span.data() == data);
  }

  SECTION("we handle multiple outputs") {
    output_bit_table = {1, 8};
    output_lengths = {1, 2};
    auto span = make_scalars_span(data, output_bit_table, output_lengths);
    REQUIRE(span.size() == 4);
    REQUIRE(span.data() == data);
  }

  SECTION("we handle values that would overflow a 32-bit integer") {
    output_bit_table = {1, 1};
    output_lengths = {
        4'294'967'295u,
        4'294'967'295u,
    };
    auto span = make_scalars_span(data, output_bit_table, output_lengths);
    REQUIRE(span.size() == 4'294'967'295ul);
    REQUIRE(span.data() == data);
  }
}
