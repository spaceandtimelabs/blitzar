#include "sxt/multiexp/bucket_method/count.h"

#include <algorithm>
#include <iostream>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can count the number of entries in buckets") {
  basdv::stream stream;
  memmg::managed_array<unsigned> count_array{memr::get_managed_device_resource()};
  memmg::managed_array<uint8_t*> scalars{memr::get_managed_device_resource()};

  SECTION("we handle the case of a single entry") {
    memmg::managed_array<uint8_t> scalars1{memr::get_managed_device_resource()};
    scalars1 = {1u};
    scalars = {scalars1.data()};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars, 1, 1, 8, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255);
    std::fill(expected.begin(), expected.end(), 0);
    expected[0] = 1;
    REQUIRE(count_array == expected);
  }
}
