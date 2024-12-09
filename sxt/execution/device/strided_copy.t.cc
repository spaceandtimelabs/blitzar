#include "sxt/execution/device/strided_copy.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("todo") {
  std::vector<int> src;
  std::pmr::vector<int> dst{memr::get_managed_device_resource()};

  basdv::stream stream;

  SECTION("we can copy empty data") {
    auto fut = strided_copy_host_to_device<int>(dst, stream, src, 1, 0, 0);
    REQUIRE(fut.ready());
  }

  // single element copy
  // basic strided copy
  // strided copy as large as buffer
  // strided copy larger than a single buffer
  // strided copy larger than a single buffer x2
/* template <class T> */
/* xena::future<> strided_copy_host_to_device(basct::span<T> dst, const basdv::stream& stream, */
/*                                            basct::cspan<T> src, size_t stride, size_t slice_size, */
/*                                            size_t offset) noexcept { */
}
