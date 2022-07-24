#include "sxt/base/container/span_void.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we track a span of unknown elements") {
  SECTION("we start empty") {
    span_void s;
    REQUIRE(s.empty());
  }

  SECTION("span_void is convertible to span_cvoid") {
    int array[] = {1, 2, 3};
    span_void s{array, 3, sizeof(int)};
    span_cvoid s2 = s;
    REQUIRE(s2.data() == array);
    REQUIRE(s2.size() == 3);
    REQUIRE(s2.element_size() == sizeof(int));
  }
}
