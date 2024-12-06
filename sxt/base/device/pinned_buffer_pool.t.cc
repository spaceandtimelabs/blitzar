#include "sxt/base/device/pinned_buffer_pool.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can pool pinned buffers") {
  size_t num_buffers = 3u;
  pinned_buffer_pool pool{num_buffers};
  REQUIRE(pool.size() == num_buffers);

  SECTION("we can aquire and release buffers") {
    auto h = pool.aquire_handle();
    REQUIRE(pool.size() == num_buffers - 1);
    pool.release_handle(h);
    REQUIRE(pool.size() == num_buffers);
  }

  SECTION("we can aquire a handle from an empty pool") {
    pinned_buffer_pool empty_pool{0};
    auto h = empty_pool.aquire_handle();
    empty_pool.release_handle(h);
    REQUIRE(empty_pool.size() == 1);
  }
}
