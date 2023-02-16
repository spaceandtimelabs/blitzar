#include "sxt/algorithm/base/identity_mapper.h"

#include "sxt/algorithm/base/mapper.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::algb;

TEST_CASE("we can map a contiguous block of data") {
  SECTION("identity_mapper satisfies the mapper concept") { REQUIRE(mapper<identity_mapper<int>>); }

  SECTION("we can index a block of data") {
    int data[] = {1, 2, 3, 4};
    identity_mapper<int> mapper{data};

    REQUIRE(mapper.map_index(0) == 1);
    int x;
    mapper.map_index(x, 1);
    REQUIRE(x == 2);
  }
}
