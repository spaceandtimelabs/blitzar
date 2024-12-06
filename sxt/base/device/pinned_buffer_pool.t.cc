#include "sxt/base/device/pinned_buffer_pool.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can pool pinned buffers") {
  size_t num_buffers = 3u;
  pinned_buffer_pool pool{num_buffers};
  REQUIRE(pool.num_buffers() == num_buffers);
}
