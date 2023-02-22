#include "sxt/multiexp/multiproduct_gpu/computation_setup.h"

#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct_computation_descriptor.h"

using namespace sxt;
using namespace sxt::mtxmpg;

TEST_CASE("we can fill in the descriptor for a multiproduct computation that runs on the GPU") {
  multiproduct_computation_descriptor descriptor;

  uint64_t ignore_v = 0;

  SECTION("we handle a single product with a single entry") {
    mtxi::index_table products{
        {ignore_v, ignore_v, 0},
    };
    basct::blob_array masks(1, 1);
    masks[0][0] = 1;
    setup_multiproduct_computation(descriptor, products.cheader(), masks, 1);
    multiproduct_computation_descriptor expected{
        .num_blocks = 1,
        .max_block_size = xenk::block_size_t::v1,
        .indexes = {0},
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
    mtxi::index_table products{
        {ignore_v, ignore_v, 0},
        {ignore_v, ignore_v, 0},
    };
    basct::blob_array masks(1, 1);
    masks[0][0] = 1;
    setup_multiproduct_computation(descriptor, products.cheader(), masks, 1);
    multiproduct_computation_descriptor expected{
        .num_blocks = 2,
        .max_block_size = xenk::block_size_t::v1,
        .indexes = {0, 0},
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
    mtxi::index_table products{
        {ignore_v, ignore_v, 0},
        {ignore_v, ignore_v, 1, 2, 3, 4},
    };
    basct::blob_array masks(5, 1);
    masks[0][0] = 1;
    masks[1][0] = 1;
    masks[2][0] = 1;
    masks[3][0] = 1;
    masks[4][0] = 1;
    setup_multiproduct_computation(descriptor, products.cheader(), masks, 5);
    multiproduct_computation_descriptor expected{
        .num_blocks = 2,
        .max_block_size = xenk::block_size_t::v2,
        .indexes = {0, 1, 2, 3, 4},
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

  SECTION("we handle inactive generators") {
    mtxi::index_table products{
        {ignore_v, ignore_v, 0, 1, 2, 3},
    };
    basct::blob_array masks(5, 1);
    masks[0][0] = 1;
    masks[1][0] = 0;
    masks[2][0] = 1;
    masks[3][0] = 1;
    masks[4][0] = 1;
    setup_multiproduct_computation(descriptor, products.cheader(), masks, 4);
    multiproduct_computation_descriptor expected{
        .num_blocks = 1,
        .max_block_size = xenk::block_size_t::v2,
        .indexes = {0, 2, 3, 4},
        .block_descriptors =
            {
                {
                    .block_offset = 0,
                    .index_first = 0,
                    .n = 4,
                    .reduction_num_blocks = 1,
                    .block_size = xenk::block_size_t::v2,
                },
            },
    };
    REQUIRE(descriptor == expected);
  }
}
