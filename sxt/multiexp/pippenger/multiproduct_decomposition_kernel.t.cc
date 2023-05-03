#include "sxt/multiexp/pippenger/multiproduct_decomposition_kernel.h"

#include <algorithm>
#include <numeric>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can decompose the bits in a multi-exponentiation") {
  basdv::stream stream;
  memmg::managed_array<unsigned> block_counts{memr::get_pinned_resource()};
  memmg::managed_array<uint8_t> exponents_data{memr::get_managed_device_resource()};
  memmg::managed_array<int32_t> signed_exponents_data{memr::get_managed_device_resource()};

  SECTION("we handle the case of no 1 bits") {
    exponents_data = {0};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    auto fut = count_exponent_bits(block_counts, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size());
  }

  SECTION("we handle an exponent with a single bit") {
    exponents_data = {1};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    auto fut = count_exponent_bits(block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[0] == 1);
  }

  SECTION("we handle an signed exponent with a single bit") {
    signed_exponents_data = {-1};
    auto exponents = mtxb::to_exponent_sequence(signed_exponents_data);
    auto fut = count_exponent_bits(block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[0] == 1);
  }

  SECTION("we handle an exponent with a single bit at a non-zero offset") {
    exponents_data = {0b10};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    auto fut = count_exponent_bits(block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[1] == 1);
  }

  SECTION("we handle multiple entries in the exponent") {
    exponents_data = {0, 0b10};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    auto fut = count_exponent_bits(block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[9] == 1);
  }
}
