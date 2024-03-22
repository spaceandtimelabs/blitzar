#include "sxt/base/num/choose.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basn;

TEST_CASE("we can compute binomial coefficients") {
  REQUIRE(choose_k(3, 0) == 1);
  REQUIRE(choose_k(3, 1) == 3);
  REQUIRE(choose_k(3, 2) == 3);
  REQUIRE(choose_k(3, 3) == 1);
  /* REQUIRE(choose_k(16u, 11u) == 1); */
}

