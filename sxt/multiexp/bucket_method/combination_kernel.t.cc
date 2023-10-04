#include "sxt/multiexp/bucket_method/combination_kernel.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can combine partial bucket sums") {
  memmg::managed_array<bascrv::element97> partial_bucket_sums(memr::get_managed_device_resource());
  memmg::managed_array<bascrv::element97> bucket_sums(memr::get_managed_device_resource());

  SECTION("we handle the case of a single bucket") {
    partial_bucket_sums = {12u};
    bucket_sums = {0u};
    combine_partial_bucket_sums<<<1, 1>>>(bucket_sums.data(), partial_bucket_sums.data(), 1);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 12u);
  }

  SECTION("we handle a bucket with two sums") {
    partial_bucket_sums = {12u, 45u};
    bucket_sums = {0u};
    combine_partial_bucket_sums<<<1, 1>>>(bucket_sums.data(), partial_bucket_sums.data(), 2);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 57u);
  }

  SECTION("we handle a bucket group size > 1") {
    partial_bucket_sums = {12u, 45u, 7u, 3u};
    bucket_sums = {0, 0};
    combine_partial_bucket_sums<<<1, 2>>>(bucket_sums.data(), partial_bucket_sums.data(), 2);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 19u);
    REQUIRE(bucket_sums[1] == 48u);
  }

  SECTION("we handle more than one bucket group") {
    partial_bucket_sums = {12u, 45u, 7u, 3u};
    bucket_sums = {0, 0};
    combine_partial_bucket_sums<<<2, 1>>>(bucket_sums.data(), partial_bucket_sums.data(), 2);
    basdv::synchronize_device();
    REQUIRE(bucket_sums[0] == 19u);
    REQUIRE(bucket_sums[1] == 48u);
  }
}

TEST_CASE("we can reduce bucket groups") {
}
/* template <bascrv::element T, unsigned BucketGroupSize, unsigned NumBucketGroups> */
/* __global__ void combine_bucket_groups(T* out, T* bucket_sums) { */
