#include "sxt/curve21/property/curve.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::c21p;

TEST_CASE("we can determine if a point is on the curve") {
  c21t::element_p3 p{
      .X{3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      .Y{1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      .Z{1, 0, 0, 0, 0},
      .T{1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  REQUIRE(is_on_curve(p));

  auto p2 = p;
  p2.X[0] += 1;
  REQUIRE(!is_on_curve(p2));
}
