#include "sxt/base/container/span.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
using namespace sxt::basct;

TEST_CASE("span represents a view into a contiguous region of memory") {
  SECTION("a span is default constructed to be empty") {
    span<int> s;
    REQUIRE(s.size() == 0);
    REQUIRE(s.data() == nullptr);
    REQUIRE(s.empty());
  }

  SECTION("we can use span to access an array") {
    int data[] = {1, 2, 3, 4, 5};
    span<int> s{data, 3};
    REQUIRE(s.data() == data);
    REQUIRE(s.size() == 3);
    REQUIRE(s[0] == 1);
    REQUIRE(s[1] == 2);
    REQUIRE(s[2] == 3);
  }

  SECTION("span is implicitly constructible from containers") {
    std::vector<int> v = {1, 2, 3};
    span<int> s{v};
    REQUIRE(s.data() == v.data());
    REQUIRE(s.size() == v.size());
  }
}
