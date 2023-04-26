#include "sxt/field51/operation/sqrt.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using f51t::operator""_f51;

TEST_CASE("we can compute square roots of field elements") {
  f51t::element rt;
  auto x = 0x4_f51;
  REQUIRE(sqrt(rt, x) == 0);
  f51t::element pow2;
  f51o::mul(pow2, rt, rt);
  REQUIRE(pow2 == x);

  REQUIRE(sqrt(rt, 0x123_f51) != 0);
}
