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
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"

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

  std::vector<uint8_t> scalars(1);
  std::vector<E> res(1);

  SECTION("we can compute a multiexponentiation with a zero scalar") {
    auto fut = multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == E::identity());
  }

  SECTION("we can compute a multiexponentiation multiexponentiation with a scalar of one") {
    scalars[0] = 1;
    auto fut = multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0]);
  }

  SECTION("we can compute a multiexponentiation with a scalar of two") {
    scalars[0] = 2;
    auto fut = multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 2u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation with a scalar of three") {
    scalars[0] = 3;
    auto fut = multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u * generators[0].value);
  }

  SECTION("we can compute a multiexponentiation with two scalars") {
    scalars.resize(2);
    scalars[0] = 1;
    scalars[1] = 1;
    auto fut = multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + generators[1].value);
  }

  SECTION("we can compute a multiexponentiation with more than 16 scalars") {
    scalars.resize(17);
    scalars[0] = 1;
    scalars[16] = 1;
    auto fut = multiexponentiate<E>(res, *accessor, 1, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == generators[0].value + generators[16].value);
  }
}
