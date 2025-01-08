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
#include "sxt/multiexp/pippenger2/combine_reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can combine and reduce partial products with outputs of fixed size") {
  using E = bascrv::element97;

  unsigned element_num_bytes = 1;
  std::vector<E> partial_products;
  std::vector<E> res(1);

  SECTION("we handle no outputs") {
    res.clear();
    auto fut = combine_reduce<E>(res, element_num_bytes, partial_products);
    REQUIRE(fut.ready());
  }

  SECTION("we can combine and reduce a single element") {
    partial_products.resize(8);
    partial_products[0] = 3u;
    auto fut = combine_reduce<E>(res, element_num_bytes, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
  }

  SECTION("we can combine and reduce elements already on device") {
    partial_products.resize(8);
    partial_products[0] = 3u;
    std::pmr::vector<E> partial_products_dev{partial_products.begin(), partial_products.end(),
                                             memr::get_managed_device_resource()};
    auto fut = combine_reduce<E>(res, element_num_bytes, partial_products_dev);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
  }

#if 0
  SECTION("we can combine and reduce a single output with a reduction size of two") {
    output_bit_table = {1};
    partial_products = {3u, 4u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 7u);
  }

  SECTION("we can combine and reduce an output with a bit width of 2") {
    output_bit_table = {2};
    partial_products = {3u, 4u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 11u);
  }

  SECTION("we can combine and reduce multiple outputs") {
    output_bit_table = {1, 1};
    partial_products = {3u, 4u};
    res.resize(2);
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
    REQUIRE(res[1] == 4u);
  }

  SECTION("we can combine and reduce in chunks") {
    output_bit_table = {1, 1};
    partial_products = {3u, 4u};
    res.resize(2);
    basit::split_options split_options{
        .max_chunk_size = 1u,
        .split_factor = 2u,
    };
    auto fut = combine_reduce<E>(res, split_options, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
    REQUIRE(res[1] == 4u);
  }
}

TEST_CASE("we can combine and reduce partial products with outputs of varying size") {
  using E = bascrv::element97;

  std::vector<unsigned> output_bit_table;
  std::vector<E> partial_products;
  std::vector<E> res(1);

  SECTION("we handle no outputs") {
    res.clear();
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    REQUIRE(fut.ready());
  }

  SECTION("we can combine and reduce a single element") {
    output_bit_table = {1};
    partial_products = {3u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
  }

  SECTION("we can combine and reduce elements already on device") {
    output_bit_table = {1};
    partial_products = {3u};
    std::pmr::vector<E> partial_products_dev{partial_products.begin(), partial_products.end(),
                                             memr::get_managed_device_resource()};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products_dev);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
  }

  SECTION("we can combine and reduce a single output with a reduction size of two") {
    output_bit_table = {1};
    partial_products = {3u, 4u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 7u);
  }

  SECTION("we can combine and reduce an output with a bit width of 2") {
    output_bit_table = {2};
    partial_products = {3u, 4u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 11u);
  }

  SECTION("we can combine and reduce multiple outputs") {
    output_bit_table = {1, 1};
    partial_products = {3u, 4u};
    res.resize(2);
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
    REQUIRE(res[1] == 4u);
  }

  SECTION("we can combine and reduce in chunks") {
    output_bit_table = {1, 1};
    partial_products = {3u, 4u};
    res.resize(2);
    basit::split_options split_options{
        .max_chunk_size = 1u,
        .split_factor = 2u,
    };
    auto fut = combine_reduce<E>(res, split_options, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
    REQUIRE(res[1] == 4u);
  }
#endif
}
