#include "sxt/multiexp/pippenger/multiproduct_decomposition_kernel.h"

#include <algorithm>
#include <numeric>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/base/stream.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can decompose the bits in a multi-exponentiation") {
  xenb::stream stream;
  memmg::managed_array<unsigned> block_counts{memr::get_pinned_resource()};

  SECTION("we handle the case of no 1 bits") {
    std::vector<uint8_t> exponents_data = {0};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> indexes_p{8, memr::get_managed_device_resource()};
    auto fut = decompose_exponent_bits(indexes_p, block_counts, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size());
  }

  SECTION("we handle an exponent with a single bit") {
    std::vector<uint8_t> exponents_data = {1};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> indexes_p{8, memr::get_managed_device_resource()};
    auto fut = decompose_exponent_bits(indexes_p, block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[0] == 1);
    REQUIRE(indexes_p[0] == 0);
  }

  SECTION("we handle an exponent with a single bit at a non-zero offset") {
    std::vector<uint8_t> exponents_data = {0b10};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> indexes_p{8, memr::get_managed_device_resource()};
    auto fut = decompose_exponent_bits(indexes_p, block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[1] == 1);
    REQUIRE(indexes_p[0] == 0);
  }

  SECTION("we handle multiple entries in the exponent") {
    std::vector<uint8_t> exponents_data = {0, 0b10};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> indexes_p{exponents_data.size() * 8,
                                             memr::get_managed_device_resource()};
    auto fut = decompose_exponent_bits(indexes_p, block_counts, stream, exponents);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(std::count(block_counts.begin(), block_counts.end(), 0) == block_counts.size() - 1);
    REQUIRE(block_counts[9] == 1);
    REQUIRE(indexes_p[9] == 1);
  }
}
