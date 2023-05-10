/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/multiexp/multiproduct_gpu/computation_setup.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct_computation_descriptor.h"

using namespace sxt;
using namespace sxt::mtxmpg;

TEST_CASE("we can fill in the descriptor for a multiproduct computation that runs on the GPU") {
  multiproduct_computation_descriptor descriptor;

  SECTION("we handle a single product with a single entry") {
    memmg::managed_array<unsigned> product_sizes = {1};
    setup_multiproduct_computation(descriptor, product_sizes);
    multiproduct_computation_descriptor expected{
        .num_blocks = 1,
        .max_block_size = xenk::block_size_t::v1,
        .block_descriptors =
            {
                {
                    .block_offset = 0,
                    .index_first = 0,
                    .n = 1,
                    .reduction_num_blocks = 1,
                    .block_size = xenk::block_size_t::v1,
                },
            },
    };
    REQUIRE(descriptor == expected);
  }

  SECTION("we handle two products with a single entry") {
    memmg::managed_array<unsigned> product_sizes = {1, 1};
    setup_multiproduct_computation(descriptor, product_sizes);
    multiproduct_computation_descriptor expected{
        .num_blocks = 2,
        .max_block_size = xenk::block_size_t::v1,
        .block_descriptors =
            {
                {
                    .block_offset = 0,
                    .index_first = 0,
                    .n = 1,
                    .reduction_num_blocks = 1,
                    .block_size = xenk::block_size_t::v1,
                },
                {
                    .block_offset = 1,
                    .index_first = 1,
                    .n = 1,
                    .reduction_num_blocks = 1,
                    .block_size = xenk::block_size_t::v1,
                },
            },
    };
    REQUIRE(descriptor == expected);
  }

  SECTION("we handle two products with varying number of entries") {
    memmg::managed_array<unsigned> product_sizes = {1, 4};
    setup_multiproduct_computation(descriptor, product_sizes);
    multiproduct_computation_descriptor expected{
        .num_blocks = 2,
        .max_block_size = xenk::block_size_t::v2,
        .block_descriptors =
            {
                {
                    .block_offset = 0,
                    .index_first = 0,
                    .n = 1,
                    .reduction_num_blocks = 1,
                    .block_size = xenk::block_size_t::v1,
                },
                {
                    .block_offset = 1,
                    .index_first = 1,
                    .n = 4,
                    .reduction_num_blocks = 1,
                    .block_size = xenk::block_size_t::v2,
                },
            },
    };
    REQUIRE(descriptor == expected);
  }
}
