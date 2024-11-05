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
#include "sxt/multiexp/pippenger2/variable_length_computation.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can fill in the table of product lengths") {
  std::vector<unsigned> product_lengths_data(10);
  std::vector<unsigned> bit_widths;
  std::vector<unsigned> output_lengths;

  basct::span<unsigned> product_lengths = {product_lengths_data.data(), 1};

  SECTION("we can compute the product length of a single output of a single bit") {
    bit_widths = {1};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 5);
    REQUIRE(product_lengths.size() == 1);
    REQUIRE(product_lengths[0] == 5);
  }

  SECTION("we handle the case when output_length is less than length") {
    bit_widths = {1};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 20);
    REQUIRE(product_lengths.size() == 1);
    REQUIRE(product_lengths[0] == 10);
  }

  SECTION("we handle output of more than a single bit") {
    bit_widths = {2};
    product_lengths = {product_lengths_data.data(), 2};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 5);
    REQUIRE(product_lengths.size() == 2);
    REQUIRE(product_lengths[0] == 5);
    REQUIRE(product_lengths[1] == 5);
  }
}

TEST_CASE("we can count the number of products") {
  std::vector<unsigned> output_bit_table;

  SECTION("we can count a single output") {
    output_bit_table = {123};
    REQUIRE(count_products(output_bit_table) == 123);
  }

  SECTION("we can count entries that would overflow a 32-bit integer") {
    output_bit_table = {
        4'294'967'295,
        4'294'967'295,
    };
    REQUIRE(count_products(output_bit_table) == 8'589'934'590);
  }
}
