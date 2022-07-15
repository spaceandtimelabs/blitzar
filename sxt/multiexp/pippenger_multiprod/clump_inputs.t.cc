#include "sxt/multiexp/pippenger_multiprod/clump_inputs.h"

#include <cstdint>

#include "sxt/base/test/unit_test.h"

#include "sxt/memory/management/managed_array.h"

#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/reduction_stats.h"
#include "sxt/multiexp/pippenger_multiprod/test_driver.h"

using namespace sxt;
using namespace sxt::mtxpmp;

TEST_CASE("we can clump the inputs of a multiproduct") {
  test_driver drv;
  reduction_stats stats;

  SECTION("clumping a single entry product does nothing") {
    memmg::managed_array<uint64_t> inputs{10};
    mtxi::index_table products{{0, 0, 0}};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    clump_inputs(inputs, stats, products.header(), num_inactive_outputs, num_inactive_inputs, drv,
                 2);

    memmg::managed_array<uint64_t> expected_inputs{10};
    REQUIRE(inputs == expected_inputs);

    REQUIRE(stats.prev_num_terms == 1);
    REQUIRE(stats.num_terms == 1);

    REQUIRE(num_inactive_outputs == 1);
    REQUIRE(num_inactive_inputs == 1);

    mtxi::index_table expected_products{{0, 1, 0}};
    REQUIRE(products == expected_products);
  }

  SECTION("clumping a product with two entries adds them together") {
    memmg::managed_array<uint64_t> inputs{10, 5};
    mtxi::index_table products{{0, 0, 0, 1}};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    clump_inputs(inputs, stats, products.header(), num_inactive_outputs, num_inactive_inputs, drv,
                 2);

    REQUIRE(inputs[0] == 15);

    REQUIRE(stats.prev_num_terms == 1);
    REQUIRE(stats.num_terms == 1);

    REQUIRE(num_inactive_outputs == 1);
    REQUIRE(num_inactive_inputs == 1);

    mtxi::index_table expected_products{{0, 1, 0}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can clump a table with two rows and three elements") {
    memmg::managed_array<uint64_t> inputs{10, 5, 999};
    mtxi::index_table products{{0, 0, 0, 1}, {1, 0, 1}};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    clump_inputs(inputs, stats, products.header(), num_inactive_outputs, num_inactive_inputs, drv,
                 2);

    REQUIRE(inputs[0] == 15);
    REQUIRE(inputs[1] == 5);

    REQUIRE(stats.prev_num_terms == 2);
    REQUIRE(stats.num_terms == 2);

    REQUIRE(num_inactive_outputs == 2);
    REQUIRE(num_inactive_inputs == 2);

    mtxi::index_table expected_products{{0, 1, 0}, {1, 1, 1}};
    REQUIRE(products == expected_products);
  }

  SECTION("identical sets of two elements within a clump across rows are "
          "reindexed to the same input") {
    memmg::managed_array<uint64_t> inputs{10, 5, 20, 999, 999};
    mtxi::index_table products{{0, 0, 0, 1}, {0, 0, 0, 1, 2}};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    clump_inputs(inputs, stats, products.header(), num_inactive_outputs, num_inactive_inputs, drv,
                 2);

    REQUIRE(inputs[0] == 20);
    REQUIRE(inputs[1] == 15);

    REQUIRE(stats.prev_num_terms == 3);
    REQUIRE(stats.num_terms == 2);

    REQUIRE(num_inactive_outputs == 0);
    REQUIRE(num_inactive_inputs == 1);

    mtxi::index_table expected_products{{0, 0, 0}, {0, 1, 0, 0}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can clump with size 3") {
    memmg::managed_array<uint64_t> inputs{10, 5, 6, 20, 999};
    mtxi::index_table products{{0, 0, 0, 1}, {1, 0, 1, 3}, {2, 0, 2}};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    clump_inputs(inputs, stats, products.header(), num_inactive_outputs, num_inactive_inputs, drv,
                 3);

    REQUIRE(inputs[0] == 15);
    REQUIRE(inputs[1] == 5);
    REQUIRE(inputs[2] == 6);
    REQUIRE(inputs[3] == 20);

    REQUIRE(stats.prev_num_terms == 4);
    REQUIRE(stats.num_terms == 4);

    REQUIRE(num_inactive_outputs == 3);
    REQUIRE(num_inactive_inputs == 4);

    mtxi::index_table expected_products{{0, 1, 0}, {1, 2, 1, 3}, {2, 1, 2}};
    REQUIRE(products == expected_products);
  }
}
