#include "sxt/field51/random/element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/element.h"

using namespace sxt;
using namespace sxt::f51rn;

TEST_CASE("we can generate random ristretto points") {
  basn::fast_random_number_generator rng{1, 2};

  SECTION("we can generate random elements") {
    f51t::element e1, e2;
    generate_random_element(e1, rng);
    generate_random_element(e2, rng);
    REQUIRE(e1 != e2);
  }
}
