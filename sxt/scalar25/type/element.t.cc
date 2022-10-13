#include "sxt/scalar25/type/element.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::s25t;

TEST_CASE("element is comparable") {
  element c1{};
  element c2{1u, 2u};
  REQUIRE(c1 == c1);
  REQUIRE(c2 == c2);
  REQUIRE(c1 != c2);
}
