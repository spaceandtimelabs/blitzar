#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE(
    "we can compute the generator indexes used for the multiproduct part of the bucket method") {
  memmg::managed_array<unsigned> bucket_counts{memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> indexes{memr::get_managed_device_resource()};
  basdv::stream stream;

  SECTION("we handle the n == 0 case") {
    std::vector<const uint8_t*> scalars = {};
    auto fut =
        fill_multiproduct_indexes(bucket_counts, indexes, stream, scalars, 1, 0, 8);
    REQUIRE(fut.ready());
  }
}

TEST_CASE("we can compute the multiproduct table used for the bucket method") {
}
