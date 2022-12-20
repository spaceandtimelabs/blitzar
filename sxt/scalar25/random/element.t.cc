#include "sxt/scalar25/random/element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::s25rn;

TEST_CASE("we can generate random scalars") {
  basn::fast_random_number_generator rng{1, 2};

  SECTION("we don't generate the same scalar") {
    s25t::element x1, x2;
    generate_random_element(x1, rng);
    generate_random_element(x2, rng);
    REQUIRE(x1 != x2);
  }

  SECTION("we can generate random scalars in bulk") {
    s25t::element xx[2];
    generate_random_elements(xx, rng);
    REQUIRE(xx[0] != xx[1]);
  }
}
