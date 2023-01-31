#include "sxt/algorithm/base/identity_mapper.h"

#include "sxt/algorithm/base/mapper.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/base/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

using namespace sxt;
using namespace sxt::algb;

TEST_CASE("we can map a contiguous block of data") {
  SECTION("identity_mapper satisfies the mapper concept") { REQUIRE(mapper<identity_mapper<int>>); }

  SECTION("we can index a block of data") {
    int data[] = {1, 2, 3, 4};
    identity_mapper<int> mapper{data};

    REQUIRE(mapper.map_index(0) == 1);
    int x;
    mapper.map_index(x, 1);
    REQUIRE(x == 2);
  }

  SECTION("we can copy a identity_mapper of device memory to the host") {
    memmg::managed_array<int> a = {1, 2, 3, 4};
    memmg::managed_array<int> a_dev{a.size(), memr::get_device_resource()};
    basdv::memcpy_host_to_device(a_dev.data(), a.data(), sizeof(int) * a.size());
    identity_mapper<int> mapper_dev{a_dev.data()};
    memmg::managed_array<int> b = {9, 10, 11};
    xenb::stream stream;
    auto mapper_host = mapper_dev.async_make_host_mapper(b.data(), stream, a.size(), 1);
    basdv::synchronize_stream(stream);
    REQUIRE(mapper_host.map_index(0) == 2);
  }
}
