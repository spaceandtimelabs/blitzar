#include "sxt/multiexp/pippenger/multiproduct_decomposition_gpu.h"

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can compute the decomposition that turns a multi-exponentiation problem into a "
          "multi-product problem") {
  basdv::stream stream;
  memmg::managed_array<unsigned> indexes{123, memr::get_managed_device_resource()};

  SECTION("we handle the case of no terms") {
    std::vector<uint8_t> exponents_data;
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    REQUIRE(indexes.empty());

    memmg::managed_array<unsigned> expected_product_sizes = {0, 0, 0, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }

  SECTION("we handle the case of exponents with no bits") {
    std::vector<uint8_t> exponents_data = {0};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    REQUIRE(indexes.empty());

    memmg::managed_array<unsigned> expected_product_sizes = {0, 0, 0, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }

  SECTION("we handle a single term with a single 1 bit") {
    std::vector<uint8_t> exponents_data = {1};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();

    memmg::managed_array<unsigned> expected_indexes = {0};
    REQUIRE(indexes == expected_indexes);

    memmg::managed_array<unsigned> expected_product_sizes = {1, 0, 0, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }

  SECTION("we handle a single term with multiple bits set") {
    std::vector<uint8_t> exponents_data = {0b11};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();

    memmg::managed_array<unsigned> expected_indexes = {0, 0};
    REQUIRE(indexes == expected_indexes);

    memmg::managed_array<unsigned> expected_product_sizes = {1, 1, 0, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }

  SECTION("we handle multiple terms with a single bit set") {
    std::vector<uint8_t> exponents_data = {1, 0, 1};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();

    memmg::managed_array<unsigned> expected_indexes = {0, 2};
    REQUIRE(indexes == expected_indexes);

    memmg::managed_array<unsigned> expected_product_sizes = {2, 0, 0, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }

  SECTION("we handle multiple terms with multiple bits set") {
    std::vector<uint8_t> exponents_data = {0b10, 0, 0b11, 0b100};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();

    memmg::managed_array<unsigned> expected_indexes = {2, 0, 2, 3};
    REQUIRE(indexes == expected_indexes);

    memmg::managed_array<unsigned> expected_product_sizes = {1, 2, 1, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }

  SECTION("we handle signed terms") {
    std::vector<int8_t> exponents_data = {-1, 0, 3};
    auto exponents = mtxb::to_exponent_sequence(exponents_data);
    memmg::managed_array<unsigned> product_sizes(sizeof(exponents_data[0]) * 8u);
    auto fut = compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();

    auto sign_bit = 1u << 31;
    memmg::managed_array<unsigned> expected_indexes = {sign_bit, 2, 2};
    REQUIRE(indexes == expected_indexes);

    memmg::managed_array<unsigned> expected_product_sizes = {2, 1, 0, 0, 0, 0, 0, 0};
    REQUIRE(product_sizes == expected_product_sizes);
  }
}
