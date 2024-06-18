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
#include "sxt/multiexp/pippenger2/reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can reduce products") {
  using E = bascrv::element97;

  std::pmr::vector<E> outputs{memr::get_managed_device_resource()};
  std::pmr::vector<E> products{memr::get_managed_device_resource()};
  basdv::stream stream;

  std::pmr::vector<E> expected;

  SECTION("we handle a single element reduction") {
    outputs.resize(1);
    products.resize(1);
    products[0] = 123u;
    reduce_products<E>(outputs, stream, products);
    basdv::synchronize_stream(stream);
    expected = {123u};
    REQUIRE(outputs == expected);
  }

  SECTION("we handle a reduction with two elements") {
    outputs.resize(1);
    products.resize(2);
    products[0] = 123u;
    products[1] = 456u;
    reduce_products<E>(outputs, stream, products);
    basdv::synchronize_stream(stream);
    expected = {123u + 2u * 456u};
    REQUIRE(outputs == expected);
  }

  SECTION("we can reduce products on the host") {
    outputs.resize(1);
    products.resize(2);
    products[0] = 123u;
    products[1] = 456u;
    reduce_products<E>(outputs, products);
    expected = {123u + 2u * 456u};
    REQUIRE(outputs == expected);
  }
}

TEST_CASE("we can reduce products with a bit table") {
  using E = bascrv::element97;

  std::pmr::vector<E> outputs{memr::get_managed_device_resource()};
  std::pmr::vector<E> products{memr::get_managed_device_resource()};
  basdv::stream stream;

  std::vector<unsigned> bit_table;

  std::pmr::vector<E> expected;

  SECTION("we can reduce a single product of one bit") {
    outputs.resize(1);
    products.resize(1);
    bit_table = {1};
    products[0] = 123u;
    reduce_products<E>(outputs, stream, bit_table, products);
    basdv::synchronize_stream(stream);
    expected = {123u};
    REQUIRE(outputs == expected);
  }

  SECTION("we can reduce a single product of two bits") {
    outputs.resize(1);
    bit_table = {2};
    products = {123, 456};
    reduce_products<E>(outputs, stream, bit_table, products);
    basdv::synchronize_stream(stream);
    expected = {123u + 2u * 456u};
    REQUIRE(outputs == expected);
  }
}
