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
#include "sxt/multiexp/pippenger2/partition_product.h"

#include <random>
#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/pippenger2/constants.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the index used to lookup the precomputed sum for a partition") {
  uint8_t scalars[32] = {};

  SECTION("we handle the zero case") {
    auto index = compute_partition_index(scalars, 1, 16, 16, 0);
    REQUIRE(index == 0);
  }

  SECTION("we handle non-zero cases") {
    scalars[0] = 1;
    scalars[2] = 1;
    auto index = compute_partition_index(scalars, 1, 16, 16, 0);
    REQUIRE(index == 5);
  }

  SECTION("we handle a bit index of 2") {
    scalars[0] = 2;
    auto index = compute_partition_index(scalars, 1, 16, 16, 1);
    REQUIRE(index == 1);
  }

  SECTION("we handle n < 16") {
    scalars[0] = 1u;
    scalars[2] = 1u;
    auto index = compute_partition_index(scalars, 1, 16, 2, 0);
    REQUIRE(index == 1);
  }

  SECTION("we handle a step size of 2") {
    scalars[16] = 1;
    auto index = compute_partition_index(scalars, 2, 16, 16, 0);
    REQUIRE(index == 1u << 8u);
  }

  SECTION("we handle a bit width of 1") {
    scalars[0] = 1;
    scalars[1] = 1;
    auto index = compute_partition_index(scalars, 1, 1, 16, 0);
    REQUIRE(index == 1);
  }
  
  SECTION("we handle a bit width of 2") {
    scalars[0] = 0;
    scalars[1] = 1;
    auto index = compute_partition_index(scalars, 1, 2, 16, 0);
    REQUIRE(index == 2);
  }
}

TEST_CASE("we can compute the product of partitions") {
  using E = bascrv::element97;
  memmg::managed_array<E> products{8, memr::get_managed_device_resource()};
  std::vector<uint8_t> scalars(1);
  memmg::managed_array<E> expected(8);
  for (auto& e : expected) {
    e = 0u;
  }

  memmg::managed_array<E> partition_table((1u << 16) * 10);
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
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we can compute a multiproduct where the number of products is not a multiple of 8") {
    scalars = {1u, 3u};
    products.resize(2);
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {
        partition_table[3],
        partition_table[2],
    };
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with an offset") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, accessor, scalars, 16);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[partition_table_size_v + 1];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with two scalars") {
    scalars = {1u, 3u};
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[3];
    expected[1] = partition_table[2];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with more 16 scalars") {
    scalars.resize(16);
    scalars[0] = 1u;
    scalars[15] = 1u;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1 + (1u << 15u)].value;
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with more than 16 scalars") {
    scalars.resize(32);
    scalars[0] = 1u;
    scalars[16] = 1u;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1].value + partition_table[partition_table_size_v + 1].value;
    REQUIRE(products == expected);
  }

  SECTION("we can compute products on the host") {
    scalars.resize(32);
    scalars[0] = 1u;
    scalars[16] = 1u;
    partition_product<E>(products, accessor, scalars, 0);
    expected[0] = partition_table[1].value + partition_table[partition_table_size_v + 1].value;
    REQUIRE(products == expected);
  }
}

TEST_CASE("we can compute the product of partitions with different bit widths") {
  using E = bascrv::element97;
  memmg::managed_array<E> products{8, memr::get_managed_device_resource()};
  std::vector<uint8_t> scalars(1);
  memmg::managed_array<E> expected(8);
  for (auto& e : expected) {
    e = 0u;
  }

  memmg::managed_array<E> partition_table((1u << 2) * 10);
  std::mt19937 rng{0};
  for (unsigned i = 0; i < partition_table.size(); ++i) {
    if (i % (1u << 2) == 0) {
      partition_table[i] = 0u;
    } else {
      partition_table[i] = std::uniform_int_distribution<unsigned>{0, 96}(rng);
    }
  }
  in_memory_partition_table_accessor accessor{memmg::managed_array<E>{partition_table}, 2};

  SECTION("we handle a product with a single scalar") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we can compute a multiproduct where the number of products is not a multiple of 8") {
    scalars = {1u, 3u};
    products.resize(2);
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {
        partition_table[3],
        partition_table[2],
    };
    REQUIRE(products == expected);
  }
}
