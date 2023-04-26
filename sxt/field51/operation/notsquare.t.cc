#include "sxt/field51/operation/notsquare.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using sxt::f51t::operator""_f51;

TEST_CASE("we can detect if an element is not a square") {
  REQUIRE(notsquare(0x4_f51) == 0);
  REQUIRE(notsquare(0x123_f51) == 1);
}
