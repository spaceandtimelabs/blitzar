#include "sxt/multiexp/bucket_method/multiproduct_table_kernel.h"

#include <vector>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can compute a bucket decomposition") {
  std::pmr::vector<uint16_t> bucket_counts{memr::get_managed_device_resource()};
  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::pmr::vector<uint8_t> bytes{memr::get_managed_device_resource()};

  std::pmr::vector<uint16_t> expected_counts;
  std::pmr::vector<uint16_t> expected_indexes;

  SECTION("we handle the case of a single element of 0") {
    bucket_counts.resize(1);
    indexes.resize(1);
    bytes = {0u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts.resize(1);
    expected_indexes.resize(1);
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of a single element of 1") {
    bucket_counts.resize(1);
    indexes.resize(1);
    bytes = {1u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts = {1u};
    expected_indexes.resize(1);
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of two elements of zero") {
    bucket_counts.resize(1);
    indexes.resize(2);
    bytes = {0u, 0u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts = {0u};
    expected_indexes.resize(2);
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of two elements of zero and one") {
    bucket_counts.resize(1);
    indexes.resize(2);
    bytes = {0u, 1u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts = {1u};
    expected_indexes = {1u, 0u};
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }
}
