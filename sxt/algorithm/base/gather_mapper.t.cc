#include "sxt/algorithm/base/gather_mapper.h"

#include "sxt/algorithm/base/mapper.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::algb;

TEST_CASE("we can map contiguous indexes to gather reads") {
  SECTION("gather_mapper satisfies the mapper concept") { REQUIRE(mapper<gather_mapper<int>>); }

  SECTION("we can index a block of remapped data") {
    int data[] = {1, 2, 3, 4};
    unsigned indexes[] = {0, 2, 1, 3};
    gather_mapper<int> mapper{data, indexes};

    REQUIRE(mapper.map_index(0) == 1);
    REQUIRE(mapper.map_index(1) == 3);
    REQUIRE(mapper.map_index(2) == 2);
    int x;
    mapper.map_index(x, 1);
    REQUIRE(x == 3);
  }
}
