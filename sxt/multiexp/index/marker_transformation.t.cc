#include "sxt/multiexp/index/marker_transformation.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/partition_marker_utility.h"
#include "sxt/multiexp/index/index_table.h"
using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can transform an index table into partition markers") {
  auto consumer = [](basct::span<uint64_t>& indexes) noexcept {
    return consume_partition_marker(indexes, 2);
  };

  SECTION("we properly handle the empty table") {
    index_table tbl;
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 0);
    REQUIRE(tbl.empty());
  }

  SECTION("we correctly handle a table with a single entry") {
    index_table tbl{{10}};
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 1);
    index_table expected_tbl{{21}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we handle the case where a row shrinks in size") {
    index_table tbl{{2, 3, 4}};
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 2);
    index_table expected_tbl{{7, 9}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we correctly handle multiple rows") {
    index_table tbl{{2, 3, 4}, {10}};
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 3);
    index_table expected_tbl{{7, 9}, {21}};
    REQUIRE(tbl == expected_tbl);
  }
}
