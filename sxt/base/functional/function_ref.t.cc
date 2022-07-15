#include "sxt/base/functional/function_ref.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basf;

static int f1() noexcept { return 1; }

TEST_CASE("function_ref is a non-owning, type-erased functor") {
  function_ref<int()> f;

  SECTION("we function_ref can wrap a lambda functor") {
    f = [] { return 2; };
    REQUIRE(f() == 2);
  }

  SECTION("we function_ref can wrap a normal function") {
    f = f1;
    REQUIRE(f() == 1);
  }
}
