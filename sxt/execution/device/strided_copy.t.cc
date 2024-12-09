#include "sxt/execution/device/strided_copy.h"

#include <vector>
#include <cstddef>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("todo") {
  std::vector<uint8_t> src;
  std::pmr::vector<uint8_t> dst{memr::get_managed_device_resource()};

  basdv::stream stream;

  SECTION("we can copy empty data") {
    auto fut = strided_copy_host_to_device<uint8_t>(dst, stream, src, 1, 0, 0);
    REQUIRE(fut.ready());
  }

  SECTION("we can copy a single byte") {
    src = {123};
    dst.resize(1);
    auto fut = strided_copy_host_to_device<uint8_t>(dst, stream, src, 1, 1, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dst[0] == 123);
  }

  // basic strided copy
  // strided copy as large as buffer
  // strided copy larger than a single buffer
  // strided copy larger than a single buffer x2
/* template <class T> */
/* xena::future<> strided_copy_host_to_device(basct::span<T> dst, const basdv::stream& stream, */
/*                                            basct::cspan<T> src, size_t stride, size_t slice_size, */
/*                                            size_t offset) noexcept { */
}
