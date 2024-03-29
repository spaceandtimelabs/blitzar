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
#include "sxt/multiexp/pippenger2/partition_table.h"

#include <vector>

#include "sxt/base/bit/iteration.h"
#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute a slice of the partition table") {
  using E = bascrv::element97;
  std::vector<E> sums(1u << 16);
  std::vector<E> generators = {1u, 2u,  3u,  4u,  5u,  6u,  7u,  8u,
                               9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};
  compute_partition_table_slice(sums.data(), generators.data());
  for (unsigned i = 0; i < sums.size(); ++i) {
    auto expected = E::identity();
    basbt::for_each_bit(reinterpret_cast<uint8_t*>(&i), sizeof(i), [&](unsigned index) noexcept {
      auto g = generators[index];
      add_inplace(expected, g);
    });
    REQUIRE(sums[i] == expected);
  }
}

TEST_CASE("we can compute the full partition table") {
  using E = bascrv::element97;
  auto n = 2u;
  auto num_entries = 1u << 16u;
  std::vector<E> sums(num_entries * n);
  std::vector<E> generators(16u * n);
  for (unsigned i = 0; i < generators.size(); ++i) {
    generators[i] = i + 1u;
  }
  compute_partition_table<E>(sums, generators);
  REQUIRE(sums[1] == generators[0]);
  REQUIRE(sums[num_entries + 1] == generators[16]);
}
