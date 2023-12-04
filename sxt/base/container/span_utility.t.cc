#include "sxt/base/container/span_utility.h"

#include <numeric>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basct;

TEST_CASE("we can make winked allocations") {
  std::pmr::monotonic_buffer_resource alloc;

  {
    auto sx = sxt::basct::winked_span<int>(&alloc, 3);
    std::iota(sx.begin(), sx.end(), 0);
    REQUIRE(sx.size() == 3);
    REQUIRE(sx[0] == 0);
    REQUIRE(sx[2] == 2);
  }
}
