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
    res = {123};
    auto f = [] __device__ __host__ (double& x) noexcept {
      x *= 2;
    };
    auto fut = transform(res, f, chunk_options, res);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 246);
  }

  SECTION("we can transform two vectors") {
    res = {2};
    std::vector<double> y = {4};
    auto f = [] __device__ __host__ (double& x, double& y) noexcept {
      x = x + y;
    };
    auto fut = transform(res, f, chunk_options, res, y);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 6);
  }

  SECTION("we can split transform across multiple chunks") {
    res = {3, 5};
    auto f = [] __device__ __host__ (double& x) noexcept {
      x *= 2;
    };
    chunk_options.max_size = 1;
    auto fut = transform(res, f, chunk_options, res);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 6);
    REQUIRE(res[1] == 10);
  }
}
