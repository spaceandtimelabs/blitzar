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
