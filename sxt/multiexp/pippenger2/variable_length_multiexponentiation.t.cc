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
#include "sxt/multiexp/pippenger2/variable_length_multiexponentiation.h"

#include <random>
#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"
#include "sxt/ristretto/random/element.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute multiexponentiations with varying lengths") {
  using E = bascrv::element97;

  std::vector<E> generators(32);
  std::mt19937 rng{0};
  for (auto& g : generators) {
    g = std::uniform_int_distribution<unsigned>{0, 96}(rng);
  }

  auto accessor = make_in_memory_partition_table_accessor<E>(generators);

  std::vector<uint8_t> scalars(1);
  std::vector<E> res(1);
  std::vector<unsigned> output_bit_table(1);
  std::vector<unsigned> output_lengths(1);

  SECTION("we handle no outputs") {
    res.clear();
    output_bit_table.clear();
    output_lengths.clear();
    scalars.clear();
    auto fut =
        async_multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    REQUIRE(fut.ready());
  }

  SECTION("we handle no outputs on the host") {
    res.clear();
    output_bit_table.clear();
    output_lengths.clear();
    scalars.clear();
    multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
  }

  SECTION("we can compute a multiexponentiation of length zero") {
    output_bit_table[0] = 1;
    output_lengths[0] = 0;
    scalars[0] = 1;
    auto fut =
        async_multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == E::identity());
  }

  SECTION("we can compute a host multiexponentiation of length zero") {
    output_bit_table[0] = 1;
    output_lengths[0] = 0;
    scalars[0] = 1;
    multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    REQUIRE(res[0] == E::identity());
  }

  SECTION("we can compute a multiexponentiation for a single bit scalar") {
    output_bit_table[0] = 1;
    output_lengths[0] = 1;
    auto fut =
        async_multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == E::identity());
  }

  SECTION("we can compute a multiexponentiation of length two") {
    output_bit_table[0] = 1;
    output_lengths[0] = 2;
    scalars = {0, 1};
    auto fut =
        async_multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[1]);
  }

  SECTION("we can compute a multiexponentiation on the host") {
    output_bit_table[0] = 1;
    output_lengths[0] = 2;
    scalars = {0, 1};
    multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    REQUIRE(res[0] == generators[1]);
  }

  SECTION("we can compute a multiexponentiation with two products of different lengths") {
    output_bit_table = {1, 1};
    output_lengths = {1, 2};
    scalars = {3, 3};
    res.resize(2);
    auto fut =
        async_multiexponentiate<E>(res, *accessor, output_bit_table, output_lengths, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0]);
    REQUIRE(res[1] == generators[0].value + generators[1].value);
  }

  SECTION("we can split a multiexponentiation") {
    basit::split_options options{
        .min_chunk_size = 16u,
        .split_factor = 2,
    };
    output_bit_table[0] = 8;
    output_lengths[0] = 17;
    scalars.resize(32);
    scalars[0] = 1;
    scalars[16] = 1;
    auto fut = multiexponentiate_impl<E>(res, *accessor, output_bit_table, output_lengths, scalars,
                                         options);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + generators[16].value);
  }
}
