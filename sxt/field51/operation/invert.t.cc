#include "sxt/field51/operation/invert.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using namespace sxt::f51t;

TEST_CASE("we can invert field elements") {
  auto e = 0x123_f51;
  f51t::element ei;
  invert(ei, e);
  f51t::element res;
  mul(res, e, ei);
  REQUIRE(res == 0x1_f51);
}
