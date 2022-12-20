#include "sxt/ristretto/random/element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/property/curve.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;
using namespace sxt::rstrn;

TEST_CASE("we can generate random ristretto points") {
  basn::fast_random_number_generator rng{1, 2};

  SECTION("we can generate uncompressed elements") {
    c21t::element_p3 p1, p2;
    generate_random_element(p1, rng);
    generate_random_element(p2, rng);
    REQUIRE(p1 != p2);
    REQUIRE(c21p::is_on_curve(p1));
    REQUIRE(c21p::is_on_curve(p2));
  }

  SECTION("we can generate compressed elements") {
    rstt::compressed_element p1, p2;
    generate_random_element(p1, rng);
    generate_random_element(p2, rng);
    REQUIRE(p1 != p2);
  }

  SECTION("we can generate elements in bulk") {
    c21t::element_p3 px[2];
    generate_random_elements(px, rng);
    REQUIRE(px[0] != px[1]);
    REQUIRE(c21p::is_on_curve(px[0]));
    REQUIRE(c21p::is_on_curve(px[1]));
  }

  SECTION("we can generate compressed elements in bulk") {
    rstt::compressed_element px[2];
    generate_random_elements(px, rng);
    REQUIRE(px[0] != px[1]);
  }
}
