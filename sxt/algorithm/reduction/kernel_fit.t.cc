#include "sxt/algorithm/reduction/kernel_fit.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/kernel/kernel_dims.h"

using namespace sxt;
using namespace sxt::algr;

TEST_CASE("we can determine the dimensions of a reduction kernel") {
  SECTION("small reductions run on the host") {
    auto dims = fit_reduction_kernel(1);
    REQUIRE(dims.num_blocks == 0);
    dims = fit_reduction_kernel(63);
    REQUIRE(dims.num_blocks == 0);
  }

  SECTION("for reductions on the device, all threads have work to do") {
    for (unsigned int n : {64, 65, 127, 128, 100'000, 1'000'000}) {
      auto dims = fit_reduction_kernel(n);
      REQUIRE(dims.num_blocks > 0);
      REQUIRE(dims.num_blocks * static_cast<unsigned int>(dims.block_size) * 2 <= n);
    }
  }
}
