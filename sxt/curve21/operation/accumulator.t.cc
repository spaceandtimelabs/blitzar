#include "sxt/curve21/operation/accumulator.h"

#include "sxt/algorithm/base/reducer.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"

using namespace sxt;
using namespace sxt::c21o;

TEST_CASE("accumulator reduces cuve21 elements") {
  c21t::element_p3 e1{
      .X{3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      .Y{1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      .Z{1, 0, 0, 0, 0},
      .T{1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  REQUIRE(algb::reducer<accumulator>);

  SECTION("we can accumulate curve21 elements") {
    auto expected = 2 * e1;
    accumulator::accumulate(e1, e1);
    REQUIRE(e1 == expected);
  }

  SECTION("we can accumulate volatile elements") {
    volatile auto p = 2 * e1;
    accumulator::accumulate(p, p);
    c21t::element_p3 res{
        .X{p.X},
        .Y{p.Y},
        .Z{p.Z},
        .T{p.T},
    };
    REQUIRE(res == 4 * e1);
  }
}
