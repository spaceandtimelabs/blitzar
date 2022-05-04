#include "sxt/multiexp/pippenger_multiprod/clump_inputs.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/test_driver.h"
using namespace sxt;
using namespace sxt::mtxpmp;

TEST_CASE("we can clump the inputs of a multiproduct") {
  test_driver drv;

  SECTION("clumping a single entry product does nothing") {
    memmg::managed_array<int> inputs{10};
    mtxi::index_table products{{0}};
    clump_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{10};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}};
    REQUIRE(products == expected_products);
  }

  SECTION("clumping a product with two entries adds them together") {
    memmg::managed_array<int> inputs{10, -5};
    mtxi::index_table products{{0, 1}};
    clump_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{5};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can clump a table with two rows and three elements") {
    memmg::managed_array<int> inputs{10, -5};
    mtxi::index_table products{{0, 1}, {1}};
    clump_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{5, -5};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}, {1}};
    REQUIRE(products == expected_products);
  }

  SECTION(
      "identical sets of two elements within a clump across rows are "
      "reindexed to the same input") {
    memmg::managed_array<int> inputs{10, -5, 20};
    mtxi::index_table products{{0, 1}, {0, 1, 2}};
    clump_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{5, 20};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}, {0, 1}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can clump with size 3") {
    memmg::managed_array<int> inputs{10, -5, 6, 20};
    mtxi::index_table products{{0, 1}, {1, 3}, {2}};
    clump_inputs(inputs, products, drv, 3);

    memmg::managed_array<int> expected_inputs{5, -5, 6, 20};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}, {1, 3}, {2}};
    REQUIRE(products == expected_products);
  }
}
