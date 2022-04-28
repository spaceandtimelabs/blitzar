#include "sxt/multiexp/index/partition.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"
using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can transform an index table into partition markers") {
  SECTION("we properly handle the empty table") {
    index_table tbl;
    REQUIRE(partition_rows(tbl.header(), 2) == 0);
    REQUIRE(tbl.empty());
  }

  SECTION("we correctly handle a table with a single entry") {
    index_table tbl{{10}};
    REQUIRE(partition_rows(tbl.header(), 2) == 1);
    index_table expected_tbl{{21}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we handle the case where a row shrinks in size") {
    index_table tbl{{2, 3, 4}};
    REQUIRE(partition_rows(tbl.header(), 2) == 2);
    index_table expected_tbl{{7, 9}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we correclty handle multiple rows") {
    index_table tbl{{2, 3, 4}, {10}};
    REQUIRE(partition_rows(tbl.header(), 2) == 3);
    index_table expected_tbl{{7, 9}, {21}};
    REQUIRE(tbl == expected_tbl);
  }
}
