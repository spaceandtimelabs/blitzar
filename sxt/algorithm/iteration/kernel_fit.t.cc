#include "sxt/algorithm/iteration/kernel_fit.h"

#include <iostream>

#include "sxt/base/num/divide_up.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/kernel/kernel_dims.h"

using namespace sxt;
using namespace sxt::algi;

TEST_CASE("we determine valid kernel dimensions for iterations") {
  for (unsigned n : {1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 100, 1'000, 10'000, 100'000, 1'000'000}) {
    auto dims = fit_iteration_kernel(n);
    auto num_iters = basn::divide_up(n, static_cast<unsigned>(dims.block_size) * dims.num_blocks) *
                     static_cast<unsigned>(dims.block_size);
    REQUIRE(dims.num_blocks > 0);
    REQUIRE((dims.num_blocks - 1) * num_iters < n);
    REQUIRE(n <= dims.num_blocks * num_iters);
  }
}
