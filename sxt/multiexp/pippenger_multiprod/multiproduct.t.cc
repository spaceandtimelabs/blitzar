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
#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct_params.h"
#include "sxt/multiexp/pippenger_multiprod/test_driver.h"
#include "sxt/multiexp/random/int_generation.h"
#include "sxt/multiexp/random/random_multiproduct_descriptor.h"
#include "sxt/multiexp/random/random_multiproduct_generation.h"
#include "sxt/multiexp/test/add_ints.h"

using namespace sxt;
using namespace sxt::mtxpmp;

static void verify_random_example(std::mt19937& rng,
                                  const mtxrn::random_multiproduct_descriptor& descriptor) {
  test_driver drv;
  mtxi::index_table products;
  size_t num_inputs;
  size_t num_entries;
  mtxrn::generate_random_multiproduct(products, num_inputs, num_entries, rng, descriptor);
  memmg::managed_array<uint64_t> inout(num_entries);
  mtxrn::generate_uint64s(basct::span<uint64_t>{inout.data(), num_inputs}, rng);
  memmg::managed_array<uint64_t> expected_result(products.num_rows());
  mtxtst::add_ints(expected_result, products.cheader(), inout);
  compute_multiproduct(inout, products, drv, num_inputs);
  for (size_t index = 0; index < products.num_rows(); ++index) {
    REQUIRE(inout[index] == expected_result[index]);
  }
}

TEST_CASE("we can compute multiproducts") {
  test_driver drv;

  SECTION("we handle the empty case") {
    memmg::managed_array<uint64_t> inout;
    mtxi::index_table products;
    compute_multiproduct(inout, products, drv, 0);
    REQUIRE(inout.empty());
  }

  SECTION("we handle a multiproduct with a single term") {
    memmg::managed_array<uint64_t> inout = {22};
    mtxi::index_table products{{0}};
    compute_multiproduct(inout, products, drv, 1);
    memmg::managed_array<uint64_t> expected_result = {22};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle a single output with multiple terms") {
    memmg::managed_array<uint64_t> inout = {22, 3};
    mtxi::index_table products{{0, 1}};
    compute_multiproduct(inout, products, drv, 2);
    REQUIRE(inout[0] == 25);
  }

  SECTION("we handle a multi-product with two outputs") {
    memmg::managed_array<uint64_t> inout = {22, 3, 10, 999, 999};
    mtxi::index_table products{{0, 1, 2}, {0, 2}};
    compute_multiproduct(inout, products, drv, 3);
    REQUIRE(inout[0] == 35);
    REQUIRE(inout[1] == 32);
  }

  SECTION("we can handle random multiproducts with only two rows and few entries") {
    std::mt19937 rng{0};
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 4,
        .min_num_sequences = 2,
        .max_num_sequences = 2,
        .max_num_inputs = 8,
    };
    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with two rows and many entries") {
    std::mt19937 rng{0};
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 20,
        .min_num_sequences = 2,
        .max_num_sequences = 2,
        .max_num_inputs = 20,
    };
    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with three rows") {
    std::mt19937 rng{0};
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 20,
        .min_num_sequences = 3,
        .max_num_sequences = 3,
        .max_num_inputs = 20,
    };
    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with multiple rows") {
    std::mt19937 rng{0};
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 20,
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .max_num_inputs = 20,
    };
    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with many rows") {
    std::mt19937 rng{0};
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_num_sequences = 100,
        .max_num_sequences = 1000,
        .max_num_inputs = 200,
    };
    for (int i = 0; i < 10; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with many inputs") {
    std::mt19937 rng{0};
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1000,
        .max_sequence_length = 2000,
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .max_num_inputs = 4000,
    };
    for (int i = 0; i < 10; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }
}
