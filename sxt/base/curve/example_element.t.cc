#include "sxt/base/curve/example_element.h"

#include "sxt/base/curve/element.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::bascrv;

TEST_CASE("element97 satifies the curve concept") {
  REQUIRE(bascrv::element<element97>);
}
