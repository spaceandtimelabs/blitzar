#include "sxt/multiexp/pippenger_multiprod/clump_outputs.h"

#include "sxt/multiexp/index/index_table.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpmp;

TEST_CASE("we compute a clumped output table") {
  mtxi::index_table table_p;
  std::vector<uint64_t> output_clumps;

  SECTION("we can clump a table with a single entry") {
    mtxi::index_table table{{0, 0, 0}};
    REQUIRE(!compute_clumped_output_table(table_p, output_clumps, table.cheader(), 1, 2));
  }

  SECTION("we can clump a table with a single row") {
    mtxi::index_table table{{0, 0, 0, 1}};
    REQUIRE(!compute_clumped_output_table(table_p, output_clumps, table.cheader(), 2, 2));
  }

  SECTION("we can clump a table with two rows") {
    mtxi::index_table table{{0, 0, 0}, {0, 0, 0}};
    REQUIRE(!compute_clumped_output_table(table_p, output_clumps, table.cheader(), 1, 2));
  }

  SECTION("we can clump a table with two rows that result in two clumps") {
    mtxi::index_table table{{0, 0, 0, 1}, {1, 0, 0}};
    REQUIRE(compute_clumped_output_table(table_p, output_clumps, table.cheader(), 2, 2));
    REQUIRE(table_p == mtxi::index_table{{0, 0, 1}, {1, 0, 0}});
    REQUIRE(output_clumps == std::vector<uint64_t>{0, 1});
  }

  SECTION("we clump a table with multiple output clumps") {
    mtxi::index_table table{
        {0, 0, 0, 2, 3, 4, 5, 6},     // 0
        {1, 0, 0, 1, 2, 4},           // 1
        {2, 0, 0, 3, 4, 5, 6, 7},     // 2
        {3, 0, 1, 2, 3, 4, 6, 7},     // 3
    };
    REQUIRE(compute_clumped_output_table(table_p, output_clumps, table.cheader(), 8, 4));
    mtxi::index_table expected_table{
        {0, 0, 0, 2, 4},  // 0
        {1, 0, 3, 5, 6},  // 1
        {2, 0, 1},        // 2
        {3, 0, 0},        // 3
        {4, 0, 4, 7},     // 4
        {5, 0, 2, 3, 6},  // 5
    };
    REQUIRE(table_p == expected_table);
    REQUIRE(output_clumps == std::vector<uint64_t>{1, 2, 6, 7, 8, 9});
  }
}

TEST_CASE("we can rewrite the multiproduct table after clumping the outputs") {
  SECTION("we handle the case of a single clump") {
    mtxi::index_table table{{0, 0, 0}};
    std::vector<size_t> clumps = {0};
    rewrite_multiproducts_with_output_clumps(table.header(), clumps, 2);
    REQUIRE(table == mtxi::index_table{{0, 0, 0}});
  }

  SECTION("we handle a single clump with two outputs") {
    mtxi::index_table table{{0, 0, 0}, {1, 0, 1}};
    std::vector<size_t> clumps = {1};
    rewrite_multiproducts_with_output_clumps(table.header(), clumps, 2);
    REQUIRE(table == mtxi::index_table{{0, 0, 0}, {1, 0, 0}});
  }

  SECTION("we handle multiple outputs with multiple clumps") {
    mtxi::index_table table{
        {0, 0, 0, 1}, {1, 0, 1, 3}, {2, 0, 2, 3, 5}, {3, 1, 0, 0, 1, 4}};
    std::vector<size_t> clumps = {1, 2, 6, 7, 8, 9};
    rewrite_multiproducts_with_output_clumps(table.header(), clumps, 4);
    mtxi::index_table expected_table{
        {0, 0, 0, 1},        // 0
        {1, 0, 0, 2},        // 1
        {2, 0, 1, 3, 4},     // 2
        {3, 1, 0, 2, 4, 5},  // 3
    };
    REQUIRE(table == expected_table);
  }
}
