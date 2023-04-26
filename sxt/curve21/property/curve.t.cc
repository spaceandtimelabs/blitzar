#include "sxt/curve21/property/curve.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/literal.h"

using namespace sxt;
using namespace sxt::c21p;
using c21t::operator""_c21;

TEST_CASE("we can determine if a point is on the curve") {
  auto p = 0x1_c21;
  REQUIRE(is_on_curve(p));

  auto p2 = p;
  p2.X[0] += 1;
  REQUIRE(!is_on_curve(p2));
}
