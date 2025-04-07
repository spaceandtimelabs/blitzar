#include "sxt/execution/device/to_device_copier.h"

#include <numeric>
#include <vector>
#include <memory_resource>
#include "sxt/base/device/pinned_buffer_pool.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can copy memory from host to device") {
  std::pmr::vector<int> dev{memr::get_managed_device_resource()};
  std::pmr::vector<int> host;
  basdv::stream stream;

  SECTION("we can copy an empty array") {
    to_device_copier copier{dev, stream};
    auto fut = copier.copy(host);
    REQUIRE(fut.ready());
  }

  SECTION("we can a single integer") {
    dev.resize(1);
    to_device_copier copier{dev, stream};

    host = {123};
    auto fut = copier.copy(host);
    REQUIRE(!fut.ready());
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }

  SECTION("we can copy two integers in to two passes") {
    dev.resize(2);
    to_device_copier copier{dev, stream};

    host = {123};
    auto fut = copier.copy(host);
    REQUIRE(fut.ready());

    host = {456};
    fut = copier.copy(host);
    REQUIRE(!fut.ready());

    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    std::pmr::vector<int> expected = {123, 456};
    REQUIRE(dev == expected);
  }

  SECTION("we can copy a full buffer") {
    dev.resize(basdv::pinned_buffer_size / sizeof(int));
    to_device_copier copier{dev, stream};
    host.resize(dev.size());
    std::iota(host.begin(), host.end(), 0);
    auto fut = copier.copy(host);
    REQUIRE(!fut.ready());
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }
}
