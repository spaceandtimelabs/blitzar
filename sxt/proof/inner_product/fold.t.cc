#include "sxt/proof/inner_product/fold.h"

#include <vector>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
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
