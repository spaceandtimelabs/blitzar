#include "sxt/ristretto/type/compressed_element.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::rstt;

TEST_CASE("compressed_element is comparable") {
  compressed_element c1{};
  compressed_element c2{1, 2};
  REQUIRE(c1 == c1);
  REQUIRE(c2 == c2);
  REQUIRE(c1 != c2);
}
