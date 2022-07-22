#include "sxt/multiexp/index/transpose.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can transpose an index table") {
  index_table table_p;

  SECTION("we can transpose an empty index table") {
    index_table table{};
    REQUIRE(transpose(table_p, table.cheader(), 0) == 0);
    REQUIRE(table_p == index_table{});
  }

  SECTION("we can transpose an index table with a single entry") {
    index_table table{{0}};
    REQUIRE(transpose(table_p, table.cheader(), 1) == 0);
    REQUIRE(table_p == index_table{{0}});
  }

  SECTION("we can transpose a table with a single row") {
    index_table table{{0, 1, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4) == 3);
    REQUIRE(table_p == index_table{{0}, {0}, {0}, {0}});
  }

  SECTION("we can transpose a table with two rows") {
    index_table table{{0, 1}, {0, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4) == 3);
    REQUIRE(table_p == index_table{{0, 1}, {0}, {1}, {1}});
  }

  SECTION("we can add padding while transposing") {
    index_table table{{0, 1}, {0, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4, 2) == 3);
    REQUIRE(table_p == index_table{{0, 0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}});
  }

  SECTION("we can transpose a table with an empty row") {
    index_table table{{0, 1}, {}, {0, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4) == 2);
    REQUIRE(table_p == index_table{{0, 2}, {0}, {2}, {2}});
  }

  SECTION("we can transpose an arbitrary table") {
    index_table table{{0, 1}, {0, 2, 3}, {2, 3, 4}, {0}};
    REQUIRE(transpose(table_p, table.cheader(), 5) == 5);
    REQUIRE(table_p == index_table{{0, 1, 3}, {0}, {1, 2}, {1, 2}, {2}});
  }
}
