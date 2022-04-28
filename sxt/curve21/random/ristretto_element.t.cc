#include "sxt/curve21/random/ristretto_element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/property/curve.h"
#include "sxt/curve21/type/element_p3.h"
using namespace sxt;
using namespace sxt::c21rn;

TEST_CASE("we can generate random ristretto points") {
  basn::fast_random_number_generator rng{1, 2};
  c21t::element_p3 p1, p2;
  generate_random_ristretto_element(p1, rng);
  generate_random_ristretto_element(p2, rng);
  REQUIRE(p1 != p2);
  REQUIRE(c21p::is_on_curve(p1));
  REQUIRE(c21p::is_on_curve(p2));
}
