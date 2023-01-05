#include "sxt/ristretto/operation/overload.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/ristretto/type/literal.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::rstt;
using sxt::s25t::operator""_s25;

TEST_CASE("we can use operators on ristretto elements") {
  auto e1 = 0x123_crs;

  SECTION("we do basic operations") {
    auto e1x2 = e1 + e1;
    REQUIRE(e1x2 != e1);
    REQUIRE(e1x2 == 0x2_s25 * e1);
    REQUIRE(e1 + (e1 - e1) == e1);
    REQUIRE(e1 + (e1 + -e1) == e1);
    REQUIRE(-(-e1) == e1);
  }

  SECTION("we can use +=") {
    auto e1p = e1;
    e1p += e1;
    REQUIRE(e1p == 0x2_s25 * e1);
  }

  SECTION("we can use -=") {
    auto e1p = e1;
    e1p -= e1;
    REQUIRE(e1p + e1 == e1);
  }
}
