#include "sxt/algorithm/iteration/transform.h"

#include <vector>

#include "sxt/execution/schedule/scheduler.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::algi;

TEST_CASE("t") {
  std::vector<double> res;
  basit::chunk_options chunk_options;

  SECTION("we can transform a vector with a single element") {
    res.resize(1);
    res[0] = 123;
    auto f = [] __device__ __host__ (double& x) noexcept {
      x *= 2;
    };
    auto fut = transform(res, f, chunk_options, res);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 246);
  }
}
