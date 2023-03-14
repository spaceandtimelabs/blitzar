#include "sxt/base/concept/memcpyable_ranges.h"

#include <list>
#include <span>
#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::bascpt;

TEST_CASE("we can distingish types for ranges that are memcpyable") {
  SECTION("we handle two vectors") {
    REQUIRE(memcpyable_ranges<std::vector<int>&, std::vector<int>>);
  }

  SECTION("ranges with different types aren't memcpyable") {
    REQUIRE(!memcpyable_ranges<std::vector<double>&, std::vector<int>>);
  }

  SECTION("non-contiguous containers aren't memcpyable") {
    REQUIRE(!memcpyable_ranges<std::vector<int>&, std::list<int>>);
  }

  SECTION("we handle spans") { REQUIRE(memcpyable_ranges<std::span<int>, std::vector<int>>); }
}
