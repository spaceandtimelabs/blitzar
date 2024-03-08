#include "sxt/multiexp/pippenger2/partition_index_kernel.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the partition indexes for a multiexponentiation") {
  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::pmr::vector<uint8_t> scalars{memr::get_managed_device_resource()};

  indexes.resize(256);
  scalars.resize(32);

  std::pmr::vector<uint16_t> expected(indexes.size());

  basdv::stream stream;

  SECTION("we handle a single scalar of zero") {
    launch_fill_partition_indexes_kernel(indexes.data(), stream, scalars.data(), 1, 1);
    basdv::synchronize_stream(stream);
    REQUIRE(indexes == expected);
  }

  SECTION("we handle a single scalar of one") {
    scalars[0] = 1;
    launch_fill_partition_indexes_kernel(indexes.data(), stream, scalars.data(), 1, 1);
    basdv::synchronize_stream(stream);
    expected[0] = 1;
    REQUIRE(indexes == expected);
  }
}
