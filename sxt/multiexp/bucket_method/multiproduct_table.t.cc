#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include <vector>
#include <iostream>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/bucket_method/bucket_descriptor.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE(
    "we can compute the generator indexes used for the multiproduct part of the bucket method") {
  memmg::managed_array<unsigned> bucket_counts{memr::get_managed_device_resource()};
  memmg::managed_array<bucket_descriptor> bucket_descriptors{memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> indexes{memr::get_managed_device_resource()};
  basdv::stream stream;

  SECTION("we handle the n == 0 case") {
    std::vector<const uint8_t*> scalars = {};
    auto fut = fill_multiproduct_indexes(bucket_counts, bucket_descriptors, indexes, stream,
                                         scalars, 1, 0, 8);
    REQUIRE(fut.ready());
  }

  SECTION("we handle the n == 1 case with a scalar of 0") {
    std::vector<uint8_t> scalars1 = {0u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, bucket_descriptors, indexes, stream,
                                         scalars, 1, 1, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    REQUIRE(indexes.empty());
    memmg::managed_array<bucket_descriptor> expected_descriptors(255u);
    for (unsigned i = 0; i < 255u; ++i) {
      expected_descriptors[i] = {
          .num_entries = 0,
          .bucket_index = i,
          .entry_first = 0,
      };
    }
    REQUIRE(bucket_descriptors == expected_descriptors);
  }

  SECTION("we handle the n == 1 case with a scalar of 1") {
    std::vector<uint8_t> scalars1 = {1u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, bucket_descriptors, indexes, stream,
                                         scalars, 1, 1, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected_indexes = {0};
    REQUIRE(indexes == expected_indexes);
    memmg::managed_array<bucket_descriptor> expected_descriptors(255u);
    expected_descriptors[0] = {
          .num_entries = 1,
          .bucket_index = 0,
          .entry_first = 0,
    };
    for (unsigned i = 1; i < 255u; ++i) {
      expected_descriptors[i] = {
          .num_entries = 0,
          .bucket_index = i,
          .entry_first = 1,
      };
    }
    REQUIRE(bucket_descriptors == expected_descriptors);
  }

  SECTION("we handle an n == 2 case with the same scalar") {
    std::vector<uint8_t> scalars1 = {1u, 1u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, bucket_descriptors, indexes, stream,
                                         scalars, 1, 2, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected_indexes = {0, 1};
    REQUIRE(indexes == expected_indexes);
    memmg::managed_array<bucket_descriptor> expected_descriptors(255u);
    expected_descriptors[0] = {
          .num_entries = 2,
          .bucket_index = 0,
          .entry_first = 0,
    };
    for (unsigned i = 1; i < 255u; ++i) {
      expected_descriptors[i] = {
          .num_entries = 0,
          .bucket_index = i,
          .entry_first = 2,
      };
    }
    REQUIRE(bucket_descriptors == expected_descriptors);
  }

  SECTION("we handle an n == 2 case with zero scalars") {
    std::vector<uint8_t> scalars1 = {0u, 1u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = fill_multiproduct_indexes(bucket_counts, bucket_descriptors, indexes, stream,
                                         scalars, 1, 2, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected_indexes = {1};
    REQUIRE(indexes == expected_indexes);
    memmg::managed_array<bucket_descriptor> expected_descriptors(255u);
    expected_descriptors[0] = {
          .num_entries = 1,
          .bucket_index = 0,
          .entry_first = 0,
    };
    for (unsigned i = 1; i < 255u; ++i) {
      expected_descriptors[i] = {
          .num_entries = 0,
          .bucket_index = i,
          .entry_first = 1,
      };
    }
    REQUIRE(bucket_descriptors == expected_descriptors);
  }
}

TEST_CASE("we can compute the multiproduct table used for the bucket method") {
  memmg::managed_array<bucket_descriptor> table{memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> indexes{memr::get_managed_device_resource()};

  SECTION("we can compute the multiproduct table for the bucket method") {
    std::vector<uint8_t> scalars1 = {2u, 1u, 2u};
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = sxt::mtxbk::compute_multiproduct_table(table, indexes, scalars, 1, 3, 8);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    memmg::managed_array<unsigned> expected_indexes = {1, 0, 2};
    REQUIRE(indexes == expected_indexes);
    memmg::managed_array<bucket_descriptor> expected_table(255u);
    expected_table[253] = {
          .num_entries = 1,
          .bucket_index = 0,
          .entry_first = 0,
    };
    expected_table[254] = {
          .num_entries = 2,
          .bucket_index = 1,
          .entry_first = 1,
    };
    for (unsigned i = 2; i < 255u; ++i) {
      expected_table[i - 2] = {
          .num_entries = 0,
          .bucket_index = i,
          .entry_first = 3,
      };
    }
    REQUIRE(table == expected_table);
  }
}