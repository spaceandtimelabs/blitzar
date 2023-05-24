/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/pippenger_multiprod/product_table_normalization.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxpmp;

TEST_CASE("we can convert a table of products into normalized form") {
  SECTION("we can normalize an empty table") {
    mtxi::index_table products;
    normalize_product_table(products, 0);
    REQUIRE(products.empty());
  }

  SECTION("we can normalize a table with a single element") {
    mtxi::index_table products{{0}};
    normalize_product_table(products, 1);
    mtxi::index_table expected_products{{0, 0, 0}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can normalize a table with a single row with multiple entries") {
    mtxi::index_table products{{0, 1, 2}};
    normalize_product_table(products, 3);
    mtxi::index_table expected_products{{0, 0, 0, 1, 2}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can normalize a table with multiple rows with multiple entries") {
    mtxi::index_table products{{0, 1, 2}, {1, 2}};
    normalize_product_table(products, 5);
    mtxi::index_table expected_products{{0, 0, 0, 1, 2}, {1, 0, 1, 2}};
    REQUIRE(products == expected_products);
  }
}
