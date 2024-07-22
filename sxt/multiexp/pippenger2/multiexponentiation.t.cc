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
#include "sxt/multiexp/pippenger2/multiexponentiation.h"

#include <random>
#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/field51/type/literal.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"
#include "sxt/ristretto/random/element.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute multiexponentiations using a precomputed table of partition sums") {
  using E = bascrv::element97;

  std::vector<E> generators(32);
  std::mt19937 rng{0};
  for (auto& g : generators) {
    g = std::uniform_int_distribution<unsigned>{0, 96}(rng);
  }

  auto accessor = make_in_memory_partition_table_accessor<E>(generators);
  auto accessor1 = make_in_memory_partition_table_accessor<E>(generators, {}, 1);
  auto accessor10 = make_in_memory_partition_table_accessor<E>(generators, {}, 10);
  (void)accessor1;
  (void)accessor10;

  std::vector<uint8_t> scalars(1);
  std::vector<E> res(1);

  SECTION("we can compute a multiexponentiation with a zero scalar") {
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == E::identity());
  }

  SECTION("we can compute a multiexponentiation multiexponentiation with a scalar of one") {
    scalars[0] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0]);
  }

  SECTION("we can compute a multiexponentiation with a scalar of two") {
    scalars[0] = 2;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation with a scalar of three") {
    scalars[0] = 3;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation with two scalars") {
    scalars.resize(2);
    scalars[0] = 1;
    scalars[1] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + generators[1].value);
  }

  SECTION("we can compute a multiexponentiation with more than 16 scalars") {
    scalars.resize(17);
    scalars[0] = 1;
    scalars[16] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + generators[16].value);
  }

  SECTION("we can compute a multiexponentiation with different window widths") {
    scalars.resize(17);
    scalars[0] = 1;
    scalars[16] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor1, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    E expected = generators[0].value + generators[16].value;
    REQUIRE(res[0] == expected);
  }

  SECTION("we can compute a multiexponentiation with more than one output") {
    res.resize(2);
    scalars.resize(2);
    scalars[0] = 1u;
    scalars[1] = 2u;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value);
    REQUIRE(res[1] == 2u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation on the host") {
    res.resize(2);
    scalars.resize(2);
    scalars[0] = 1u;
    scalars[1] = 2u;
    multiexponentiate<E>(res, *accessor, 1, scalars);
    REQUIRE(res[0] == generators[0].value);
    REQUIRE(res[1] == 2u * generators[0].value);
  }

  SECTION("we can split a multi-exponentiation") {
    multiexponentiate_options options{
        .split_factor = 2,
        .min_chunk_size = 16u,
    };
    scalars.resize(32);
    scalars[0] = 1;
    scalars[16] = 1;
    auto fut = multiexponentiate_impl<E>(res, *accessor, 1, scalars, options);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + generators[16].value);
  }

  SECTION("we can split a multi-exponentiation with more than one output") {
    multiexponentiate_options options{
        .split_factor = 2,
        .min_chunk_size = 16u,
    };
    scalars.resize(64);
    scalars[0] = 1;
    scalars[1] = 2;
    scalars[32] = 3;
    scalars[33] = 4;
    res.resize(2);
    auto fut = multiexponentiate_impl<E>(res, *accessor, 1, scalars, options);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + 3 * generators[16].value);
    REQUIRE(res[1] == 2 * generators[0].value + 4 * generators[16].value);
  }
}

TEST_CASE("we can compute multiexponentiations with packed scalars") {
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

  SECTION("we can compute a multiexponentiation for a single bit scalar") {
    output_bit_table[0] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor, output_bit_table, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == E::identity());
  }

  SECTION("we can compute a multiexponentiation for a two bit scalar") {
    output_bit_table[0] = 2;
    scalars[0] = 2u;
    auto fut = async_multiexponentiate<E>(res, *accessor, output_bit_table, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation with multiple outputs of varying bit sizes") {
    output_bit_table = {2, 1, 3};
    scalars = {0b110011};
    res.resize(3);
    auto fut = async_multiexponentiate<E>(res, *accessor, output_bit_table, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u * generators[0].value);
    REQUIRE(res[1] == 0u);
    REQUIRE(res[2] == 6u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation with multiple outputs of varying bit sizes and "
          "length 2") {
    output_bit_table = {2, 1, 3};
    scalars = {0b110011, 0b101101};
    res.resize(3);
    auto fut = async_multiexponentiate<E>(res, *accessor, output_bit_table, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u * generators[0].value + generators[1].value);
    REQUIRE(res[1] == generators[1].value);
    REQUIRE(res[2] == 6u * generators[0].value + 5u * generators[1].value);
  }

  SECTION("we can compute packed multiexponentiations on the host") {
    output_bit_table = {2, 1, 3};
    scalars = {0b110011, 0b101101};
    res.resize(3);
    multiexponentiate<E>(res, *accessor, output_bit_table, scalars);
    REQUIRE(res[0] == 3u * generators[0].value + generators[1].value);
    REQUIRE(res[1] == generators[1].value);
    REQUIRE(res[2] == 6u * generators[0].value + 5u * generators[1].value);
  }
}

TEST_CASE("we can compute multiexponentiations with curve-21") {
  using E = c21t::element_p3;
  using Ep = c21t::compact_element;

  std::vector<E> generators(32);
  basn::fast_random_number_generator rng{1, 2};
  for (auto& g : generators) {
    rstrn::generate_random_element(g, rng);
  }

  auto accessor = make_in_memory_partition_table_accessor<E>(generators);
  auto accessor_p = make_in_memory_partition_table_accessor<Ep, E>(generators);

  std::vector<uint8_t> scalars(1);
  std::vector<E> res(1);

  SECTION("we can compute a multiexponentiation multiexponentiation with a scalar of one") {
    scalars[0] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0]);
  }

  SECTION("we can compute a multi-exponentiation with an accessor using a different curve form") {
    scalars[0] = 1;
    auto fut = async_multiexponentiate<E>(res, *accessor_p, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0]);
  }
}
