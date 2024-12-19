#include "sxt/proof/sumcheck/mle_utility.h"

#include <vector>

#include "sxt/scalar25/type/element.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::prfsk;

TEST_CASE("we can query the fraction of device memory taken by MLEs") {
  std::vector<s25t::element> mles;

  SECTION("we handle the zero case") {
    REQUIRE(get_gpu_memory_fraction(mles) == 0.0);
  }

  SECTION("the fractions doubles if the length of mles doubles") {
    mles.resize(1);
    auto f1 = get_gpu_memory_fraction(mles);
    REQUIRE(f1 > 0);
    mles.resize(2);
    auto f2 = get_gpu_memory_fraction(mles);
    REQUIRE(f2 == Catch::Approx(2 * f1));
  }
}
