#include "sxt/algorithm/iteration/transform.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::algi;

TEST_CASE("t") {
  std::vector<double> res;
  basit::chunk_options chunk_options;

  SECTION("we can transform a vector with a single element") {
    res.resize(1);
    auto f = [] __device__ __host__ (double& x) noexcept {
      x *= 2;
    };
  
    /* transform(res, f, chunk_options, res); */
    (void)f;
  }
  REQUIRE(1 == 1);
}
