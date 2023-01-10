#include "sxt/proof/inner_product/fold.h"

#include <vector>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::prfip;

TEST_CASE("we can fold scalars") {
  basn::fast_random_number_generator rng{1, 2};
  s25t::element x, y;
  s25rn::generate_random_element(x, rng);
  s25rn::generate_random_element(y, rng);

  SECTION("we can fold 2 scalars") {
    std::vector<s25t::element> scalars(2);
    s25rn::generate_random_elements(scalars, rng);
    std::vector<s25t::element> scalars_p(2);
    basct::span<s25t::element> res = scalars_p;
    fold_scalars(res, scalars, x, y, 1);
    scalars_p.resize(res.size());

    std::vector<s25t::element> expected = {
        x * scalars[0] + y * scalars[1],
    };
    REQUIRE(scalars_p == expected);
  }

  SECTION("we can fold 3 scalars") {
    std::vector<s25t::element> scalars(3);
    s25rn::generate_random_elements(scalars, rng);
    std::vector<s25t::element> scalars_p(2);
    basct::span<s25t::element> res = scalars_p;
    fold_scalars(res, scalars, x, y, 2);
    scalars_p.resize(res.size());

    std::vector<s25t::element> expected = {
        x * scalars[0] + y * scalars[2],
        x * scalars[1],
    };
    REQUIRE(scalars_p == expected);
  }
}

TEST_CASE("we can fold generators") {
  basn::fast_random_number_generator rng{1, 2};
  s25t::element x, y;
  s25rn::generate_random_element(x, rng);
  s25rn::generate_random_element(y, rng);

  SECTION("we can fold 2 generators") {
    std::vector<c21t::element_p3> generators(2);
    rstrn::generate_random_elements(generators, rng);
    std::vector<c21t::element_p3> generators_p(2);
    basct::span<c21t::element_p3> res = generators_p;
    fold_generators(res, generators, x, y, 1);
    generators_p.resize(res.size());

    std::vector<c21t::element_p3> expected = {
        x * generators[0] + y * generators[1],
    };
    REQUIRE(generators_p == expected);
  }

  SECTION("we can fold 4 generators") {
    std::vector<c21t::element_p3> generators(4);
    rstrn::generate_random_elements(generators, rng);
    std::vector<c21t::element_p3> generators_p(2);
    basct::span<c21t::element_p3> res = generators_p;
    fold_generators(res, generators, x, y, 2);
    generators_p.resize(res.size());

    std::vector<c21t::element_p3> expected = {
        x * generators[0] + y * generators[2],
        x * generators[1] + y * generators[3],
    };
    REQUIRE(generators_p == expected);
  }
}
