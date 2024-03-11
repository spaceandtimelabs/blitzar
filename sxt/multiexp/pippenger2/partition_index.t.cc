#include "sxt/multiexp/pippenger2/partition_index.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the partition indexes that correspond to a multiexponentiation") {

  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::vector<const uint8_t*> scalars;

  std::pmr::vector<uint16_t> expected;

  SECTION("we handle the case of a single element") {
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    indexes.resize(256u);
    auto fut = fill_partition_indexes(indexes, scalars, 32, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected.resize(indexes.size());
    REQUIRE(indexes == expected);
  }
}
