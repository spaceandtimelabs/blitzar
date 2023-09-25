#include "sxt/multiexp/bucket_method/combination_kernel.h"

#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can combine partial bucket sums") {
  memmg::managed_array<uint64_t> partial_bucket_sums(memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> bucket_sums(memr::get_managed_device_resource());

  SECTION("we handle the case of a single bucket") {
    partial_bucket_sums = {123};
    bucket_sums = {0};
    combine_partial_bucket_sums<algr::test_add_reducer>
        <<<1, 1>>>(bucket_sums.data(), partial_bucket_sums.data(), 1);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 123);
  }

  SECTION("we handle a bucket with two sums") {
    partial_bucket_sums = {123, 456};
    bucket_sums = {0};
    combine_partial_bucket_sums<algr::test_add_reducer>
        <<<1, 1>>>(bucket_sums.data(), partial_bucket_sums.data(), 2);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 579);
  }

  SECTION("we handle a bucket group size > 1") {
    partial_bucket_sums = {123, 456, 7, 3};
    bucket_sums = {0, 0};
    combine_partial_bucket_sums<algr::test_add_reducer>
        <<<1, 2>>>(bucket_sums.data(), partial_bucket_sums.data(), 2);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 130);
    REQUIRE(bucket_sums[1] == 459);
  }

  SECTION("we handle more than one bucket group") {
    partial_bucket_sums = {123, 456, 7, 3};
    bucket_sums = {0, 0};
    combine_partial_bucket_sums<algr::test_add_reducer>
        <<<2, 1>>>(bucket_sums.data(), partial_bucket_sums.data(), 2);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 130);
    REQUIRE(bucket_sums[1] == 459);
  }
}
