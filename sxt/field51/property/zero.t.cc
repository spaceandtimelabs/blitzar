#include "sxt/field51/property/zero.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/literal.h"

using namespace sxt::f51t;
using namespace sxt::f51p;

TEST_CASE("we can determine if a field element is zero") {
  REQUIRE(is_zero(0x0_f51));
  REQUIRE(!is_zero(0x1_f51));
  REQUIRE(is_zero(0x0_f51));

  // e = 2^255 - 19
  auto e = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec_f51;
  e[0] += 1;

  REQUIRE(is_zero(e));
}
