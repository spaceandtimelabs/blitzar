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
  memmg::managed_array<uint8_t> scalars{memr::get_managed_device_resource()};

  SECTION("we handle the case of a single entry of 0") {
    scalars = {0u};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars, 1, 1, 1, 8, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255);
    std::fill(expected.begin(), expected.end(), 0);
    REQUIRE(count_array == expected);
  }

  SECTION("we handle the case of a single entry of 1") {
    scalars = {1u};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars, 1, 1, 1, 8, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255);
    std::fill(expected.begin(), expected.end(), 0);
    expected[0] = 1;
    REQUIRE(count_array == expected);
  }

  SECTION("we handle the case of a single entry of 2") {
    scalars = {2u};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars, 1, 1, 1, 8, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255);
    std::fill(expected.begin(), expected.end(), 0);
    expected[1] = 1;
    REQUIRE(count_array == expected);
  }

  SECTION("we handle multiple entries") {
    scalars = {2u, 2u, 1u};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars, 1, 3, 1, 8, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255);
    std::fill(expected.begin(), expected.end(), 0);
    expected[0] = 1;
    expected[1] = 2;
    REQUIRE(count_array == expected);
  }

  SECTION("we handle multiple partitions") {
    scalars = {1u, 3u};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars,  1, 2, 1, 8, 2);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255 * 2);
    std::fill(expected.begin(), expected.end(), 0);
    expected[0 * 2] = 1;
    expected[2 * 2 + 1] = 1;
    REQUIRE(count_array == expected);
  }

  SECTION("we handle multiple outputs") {
    scalars = {1u, 3u};
    sxt::mtxbk::count_bucket_entries(count_array, stream, scalars,  1, 1, 2, 8, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected(255 * 2);
    std::fill(expected.begin(), expected.end(), 0);
    expected[0] = 1;
    expected[255 + 2] = 1;
    REQUIRE(count_array == expected);
  }
}