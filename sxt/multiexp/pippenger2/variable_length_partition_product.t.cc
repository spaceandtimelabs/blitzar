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
#include "sxt/multiexp/pippenger2/variable_length_partition_product.h"

#include <random>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute partition products of variable length") {
  using E = bascrv::element97;

  memmg::managed_array<E> products{8, memr::get_managed_device_resource()};
  std::vector<unsigned> lengths(8, 1);
  std::vector<uint8_t> scalars(1);
  memmg::managed_array<E> expected(8);
  for (auto& e : expected) {
    e = 0u;
  }

  auto partition_table_size = 1u << 16;
  memmg::managed_array<E> partition_table(partition_table_size * 10);
  std::mt19937 rng{0};
  for (unsigned i = 0; i < partition_table.size(); ++i) {
    if (i % (1u << 16u) == 0) {
      partition_table[i] = 0u;
    } else {
      partition_table[i] = std::uniform_int_distribution<unsigned>{0, 96}(rng);
    }
  }
  in_memory_partition_table_accessor accessor{memmg::managed_array<E>{partition_table}, 16};

  SECTION("we handle a product with a single scalar") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, 8, accessor, scalars, lengths, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with an offset") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, 8, accessor, scalars, lengths, 16);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[partition_table_size + 1];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with different lengths") {
    scalars[0] = 1;
    lengths[0] = 0;
    auto fut = async_partition_product<E>(products, 8, accessor, scalars, lengths, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(products == expected);
  }

  SECTION("we handle a product slice") {
    scalars[0] = 2;
    auto fut = async_partition_product<E>(basct::span<E>{products}.subspan(1), 8, accessor, scalars,
                                          basct::span<unsigned>{lengths}.subspan(1), 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[1] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product on the host") {
    scalars[0] = 2;
    partition_product<E>(basct::span<E>{products}.subspan(1), 8, accessor, scalars,
                         basct::span<unsigned>{lengths}.subspan(1), 0);
    expected[1] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we handle products with length greater than 1") {
    scalars = {1u, 3u};
    products.resize(2);
    lengths.resize(2);
    lengths[0] = 2;
    lengths[1] = 1;
    auto fut = async_partition_product<E>(products, 2, accessor, scalars, lengths, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {
        partition_table[3],
        0,
    };
    REQUIRE(products == expected);
  }
}
