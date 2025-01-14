#include "sxt/execution/device/generate.h"

#include <cstddef>
#include <numeric>
#include <vector>

#include "sxt/base/device/pinned_buffer.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can generate an array into device memory") {
  const auto bufsize = basdv::pinned_buffer::size();
  std::pmr::vector<uint8_t> dst{memr::get_managed_device_resource()};

  basdv::stream stream;

  auto f = []<class T>(basct::span<T> buffer, size_t index) noexcept {
    for (auto& x : buffer) {
      x = static_cast<T>(index++);
    }
  };

  SECTION("we can generate an empty array") {
    auto fut = generate_to_device<uint8_t>(dst, stream, f);
    REQUIRE(fut.ready());
  }

  SECTION("we can generate a single element") {
    dst.resize(1);
    auto fut = generate_to_device<uint8_t>(dst, stream, f);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dst[0] == 0);
  }

  SECTION("we can generate two elements") {
    dst.resize(2);
    auto fut = generate_to_device<uint8_t>(dst, stream, f);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dst[0] == 0);
    REQUIRE(dst[1] == 1);
  }

  SECTION("we can generate ints") {
    std::pmr::vector<int> dst_p{2, memr::get_managed_device_resource()};
    auto fut = generate_to_device<int>(dst_p, stream, f);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dst_p[0] == 0);
    REQUIRE(dst_p[1] == 1);
  }

  SECTION("we can generate elements larger than the buffersize") {
    std::pmr::vector<int> dst_p{bufsize, memr::get_managed_device_resource()};
    auto fut = generate_to_device<int>(dst_p, stream, f);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    for (int i=0; i<bufsize; ++i) {
      REQUIRE(dst_p[i] == i);
    }
  }
}
