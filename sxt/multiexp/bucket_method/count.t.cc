#include "sxt/multiexp/bucket_method/count.h"

#include <algorithm>
#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can compute the inclusive prefix sum of bucket counts") {
  unsigned element_num_bytes = 1u;
  constexpr unsigned bit_width = 8u;
  constexpr unsigned tile_size = 1024u;
  constexpr unsigned num_outputs = 1u;

  std::pmr::vector<uint8_t> bytes{memr::get_managed_device_resource()};

  constexpr auto num_buckets_per_digit = (1u << bit_width) - 1u;
  constexpr auto num_digits = 1u;
  std::pmr::vector<unsigned> counts(num_buckets_per_digit * num_digits,
                                    memr::get_managed_device_resource());

  std::pmr::vector<unsigned> expected(counts.size());

  basdv::stream stream;

  SECTION("we handle the case of a single digit of zero") {
    bytes = {0u};
    inclusive_prefix_count_buckets(counts, stream, bytes, element_num_bytes, bit_width,
                                   num_outputs, tile_size, 1);
    basdv::synchronize_stream(stream);
    REQUIRE(counts == expected);
  }

  SECTION("we handle the case of a single digit of one") {
    bytes = {1u};
    inclusive_prefix_count_buckets(counts, stream, bytes, element_num_bytes, bit_width,
                                   num_outputs, tile_size, 1);
    basdv::synchronize_stream(stream);
    std::fill(expected.begin(), expected.end(), 1);
    REQUIRE(counts == expected);
  }

  SECTION("we handle the case of two digits of one") {
    bytes = {1u, 1u};
    inclusive_prefix_count_buckets(counts, stream, bytes, element_num_bytes, bit_width,
                                   num_outputs, tile_size, 2);
    basdv::synchronize_stream(stream);
    std::fill(expected.begin(), expected.end(), 2);
    REQUIRE(counts == expected);
  }

  SECTION("we handle the case of two digits of one and two") {
    bytes = {1u, 2u};
    inclusive_prefix_count_buckets(counts, stream, bytes, element_num_bytes, bit_width,
                                   num_outputs, tile_size, 2);
    basdv::synchronize_stream(stream);
    std::fill(expected.begin(), expected.end(), 2);
    expected[0] = 1;
    REQUIRE(counts == expected);
  }
}
