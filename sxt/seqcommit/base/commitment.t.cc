#include "sxt/seqcommit/base/commitment.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt::sqcb;

TEST_CASE("commitment is comparable") {
  commitment c1{};
  commitment c2{1, 2};
  REQUIRE(c1 == c1);
  REQUIRE(c2 == c2);
  REQUIRE(c1 != c2);
}
