#include "sxt/curve21/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::c21o;

TEST_CASE("we can double curve elements") {
  c21t::element_p3 g{
      .X{3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      .Y{1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      .Z{1, 0, 0, 0, 0},
      .T{1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  SECTION("doubling gives us the same result as adding an element to itself") {
    c21t::element_p3 res, res2;

    add(res, g, g);

    c21t::element_p1p1 res2_p1p1;
    double_element(res2_p1p1, g);
    c21t::to_element_p3(res2, res2_p1p1);

    REQUIRE(res == res2);
  }
}
