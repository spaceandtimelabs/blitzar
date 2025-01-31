/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/pippenger2/multiexponentiation_serialization.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/temp_directory.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can serialize and deserialize a variable length multiexponentiation") {
  using E = bascrv::element97;

  bastst::temp_directory dir;

  std::vector<E> generators = {11, 22, 33};
  auto accessor = make_in_memory_partition_table_accessor<E>(generators);
  std::vector<unsigned> output_bit_table = {1, 1};
  std::vector<unsigned> output_lengths = {1, 2};
  std::vector<uint8_t> scalars = {3, 2, 1};

  SECTION("we can serialize then deserialize") {
    write_multiexponentiation<E>(dir.name(), *accessor, output_bit_table, output_lengths, scalars);

    variable_length_multiexponentiation_descriptor<E, E> descr;
    read_multiexponentiation(descr, dir.name());
    REQUIRE(descr.output_bit_table == output_bit_table);
    REQUIRE(descr.output_lengths == output_lengths);
    REQUIRE(descr.scalars == scalars);
    std::vector<E> generators_p(2);
    descr.accessor->copy_generators(generators_p);
    REQUIRE(generators[0] == generators_p[0]);
    REQUIRE(generators[1] == generators_p[1]);
    REQUIRE(descr.accessor->window_width() == accessor->window_width());
  }
}
