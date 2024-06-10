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
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"

#include <random>
#include <vector>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can create a partition table accessor from given generators") {
  using E = bascrv::element97;
  std::vector<E> generators(32);
  std::mt19937 rng{0};
  for (auto& g : generators) {
    g = std::uniform_int_distribution<unsigned>{0, 96}(rng);
  }

  basdv::stream stream;
  memmg::managed_array<E> table_dev{1u << 16u, memr::get_device_resource()};
  memmg::managed_array<E> table(table_dev.size());

  SECTION("we can create an accessor from a single generators") {
    auto accessor = make_in_memory_partition_table_accessor<E>(basct::subspan(generators, 0, 1));
    accessor->async_copy_to_device(table_dev, stream, 0);
    basdv::async_copy_device_to_host(table, table_dev, stream);
    basdv::synchronize_stream(stream);
    REQUIRE(table[1] == generators[0]);
    REQUIRE(table[2] == E::identity());
  }

  SECTION("we can create an accessor from multiple generators") {
    auto accessor = make_in_memory_partition_table_accessor<E>(generators);
    accessor->async_copy_to_device(table_dev, stream, 0);
    basdv::async_copy_device_to_host(table, table_dev, stream);
    basdv::synchronize_stream(stream);
    REQUIRE(table[1] == generators[0]);
    REQUIRE(table[2] == generators[1]);
    REQUIRE(table[3] == generators[0].value + generators[1].value);
  }
}
