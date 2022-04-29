#include "sxt/multiexp/pippenger_multiprod/partition_inputs.h"

#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/test_driver.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpmp;

TEST_CASE("we can partition the inputs of a multiproduct") {
  test_driver drv;

  SECTION("partitioning a single entry product does nothing") {
    memmg::managed_array<int> inputs{10};
    mtxi::index_table products{{0}};
    partition_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{10};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}};
    REQUIRE(products == expected_products);
  }

  SECTION("partitioning a product with two entries adds them together") {
    memmg::managed_array<int> inputs{10, -5};
    mtxi::index_table products{{0, 1}};
    partition_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{5};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{0}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can partition a table with two rows and three elements") {
    memmg::managed_array<int> inputs{10, -5};
    mtxi::index_table products{{0, 1}, {1}};
    partition_inputs(inputs, products, drv, 2);

    memmg::managed_array<int> expected_inputs{-5, 5};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{1}, {0}};
    REQUIRE(products == expected_products);
  }

  SECTION("we can partition with size 3") {
    memmg::managed_array<int> inputs{10, -5, 6, 20};
    mtxi::index_table products{{0, 1}, {1, 3}, {2}};
    partition_inputs(inputs, products, drv, 3);

    memmg::managed_array<int> expected_inputs{-5, 5, 6, 20};
    REQUIRE(inputs == expected_inputs);

    mtxi::index_table expected_products{{1}, {0, 3}, {2}};
    REQUIRE(products == expected_products);
  }
}
