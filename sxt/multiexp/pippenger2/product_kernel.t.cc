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
#include "sxt/multiexp/pippenger2/product_kernel.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute products with a lookup table") {
  std::pmr::vector<bascrv::element97> products{memr::get_managed_device_resource()};
  std::pmr::vector<bascrv::element97> table{memr::get_managed_device_resource()};
  std::pmr::vector<uint16_t> bitsets{memr::get_managed_device_resource()};

  constexpr auto num_elements = 1u << 16u;

  products.resize(1);
  table.resize(num_elements);
  bitsets.resize(1);

  std::pmr::vector<bascrv::element97> expected;

  basdv::stream stream;

  SECTION("we handle the case of a single partition") {
    table[0] = 123u;
    launch_product_kernel<0>(products.data(), stream, bitsets.data(), table.data(), 1, 1);
    basdv::synchronize_stream(stream);
    expected = {123u};
    REQUIRE(products == expected);
  }

  SECTION("we handle two partitions with two items per thread") {
    table.resize(num_elements * 2);
    bitsets = {1, 2};
    table[1] = 123u;
    table[num_elements + 2] = 456u;
    launch_product_kernel<1>(products.data(), stream, bitsets.data(), table.data(), 1, 2);
    basdv::synchronize_device();
    expected = {123u + 456u};
    REQUIRE(products == expected);
  }

  SECTION("we handle two partitions with one item per thread") {
    table.resize(num_elements * 2);
    bitsets = {1, 2};
    table[1] = 123u;
    table[num_elements + 2] = 456u;
    launch_product_kernel<0>(products.data(), stream, bitsets.data(), table.data(), 1, 2);
    basdv::synchronize_device();
    expected = {123u + 456u};
    REQUIRE(products == expected);
  }

  SECTION("we handle two products") {
    table[0] = 123u;
    table[1] = 456u;
    bitsets = {1, 0};
    products.resize(2);
    launch_product_kernel<0>(products.data(), stream, bitsets.data(), table.data(), 2, 1);
    basdv::synchronize_device();
    expected = {456u, 123u};
    REQUIRE(products == expected);
  }
}
