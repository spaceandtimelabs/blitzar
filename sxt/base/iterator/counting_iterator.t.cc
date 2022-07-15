#include "sxt/base/iterator/counting_iterator.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basit;

TEST_CASE("we can use counting_iterator to represent a sequence of integers") {
  SECTION("we can iterate between a range of ints") {
    std::vector<int> v{counting_iterator<int>{2}, counting_iterator<int>{5}};
    std::vector<int> expected_v = {2, 3, 4};
    REQUIRE(v == expected_v);
  }
}
