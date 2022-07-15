#include "sxt/base/container/span_void.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we track a span of unknown elements") {
  SECTION("we start empty") {
    span_void s;
    REQUIRE(s.empty());
  }
}
