#include "sxt/execution/async/continuation_fn.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::xena;

TEST_CASE("we can determine if a given functor can serve as a continuation") {
  SECTION("we handle functors that aren't noexcept") {
    auto f1 = []() {};
    REQUIRE(!continuation_fn<decltype(f1), int, int>);
    REQUIRE(!continuation_fn<decltype(f1), void, void>);
  }

  SECTION("we handle void functors") {
    auto f1 = []() noexcept {};
    REQUIRE(continuation_fn<decltype(f1), void, void>);

    auto f2 = []() noexcept { return 123; };
    REQUIRE(continuation_fn<decltype(f2), void, int>);
  }

  SECTION("we handle non-void functors") {
    auto f1 = [](int x) noexcept { return x; };
    REQUIRE(continuation_fn<decltype(f1), int, int>);
  }

  SECTION("we handle convertible functors") {
    auto f1 = [](int x) noexcept { return x; };
    REQUIRE(continuation_fn<decltype(f1), int, long>);
  }
}
