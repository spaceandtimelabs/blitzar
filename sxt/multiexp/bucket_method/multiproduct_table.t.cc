#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
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

  SECTION("we handle the n == 1 case with a scalar of 0") {
    std::vector<uint8_t> scalars1 = {0u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, indexes, stream, scalars, 1, 1, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected;
    REQUIRE(indexes == expected);
  }

  SECTION("we handle the n == 1 case with a scalar of 1") {
    std::vector<uint8_t> scalars1 = {1u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, indexes, stream, scalars, 1, 1, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0};
    REQUIRE(indexes == expected);
  }

  SECTION("we handle an n == 2 case with the same scalar") {
    std::vector<uint8_t> scalars1 = {1u, 1u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, indexes, stream, scalars, 1, 2, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0, 1};
    REQUIRE(indexes == expected);
  }

  SECTION("we handle an n == 2 case with zero scalars") {
    std::vector<uint8_t> scalars1 = {0u, 1u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, indexes, stream, scalars, 1, 2, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {1};
    REQUIRE(indexes == expected);
  }
}

TEST_CASE("we can compute the multiproduct table used for the bucket method") {
}
