#include "sxt/base/num/constexpr_switch.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basn;

TEST_CASE("we can switch to generate compile-time constants from runtime constants") {
  constexpr_switch<5>(3, [&]<unsigned I>(std::integral_constant<unsigned, I>) { REQUIRE(I == 3); });
}
