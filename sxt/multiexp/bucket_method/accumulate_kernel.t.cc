#include "sxt/multiexp/bucket_method/accumulate_kernel.h"

#include "sxt/algorithm/base/identity_mapper.h"
#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can accumulate the buckets for a multi-exponentiation") {
  memmg::managed_array<uint64_t> bucket_sums(255, memr::get_managed_device_resource());
  memmg::managed_array<uint8_t> scalars(memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> generators(memr::get_managed_device_resource());

  SECTION("we handle the empty case") {
    bucket_accumulate<algr::test_add_reducer><<<1, 1>>>(bucket_sums.data(),
                                                        algb::identity_mapper{generators.data()},
                                                        scalars.data(), 0);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      REQUIRE(bucket_sums[i] == 0);
    }
  }

  SECTION("we handle an accumulation with a single element") {
    scalars = {2};
    generators = {123};
    bucket_accumulate<algr::test_add_reducer><<<1, 1>>>(bucket_sums.data(),
                                                        algb::identity_mapper{generators.data()},
                                                        scalars.data(), 1);
    basdv::synchronize_device();
    for (unsigned i=0; i<bucket_sums.size(); ++i) {
      if (i + 1 != scalars[0]) {
        REQUIRE(bucket_sums[i] == 0);
      } else {
        REQUIRE(bucket_sums[i] == generators[0]);
      }
    }
  }

  SECTION("we ignore zero scalars") {
    scalars = {0};
    generators = {123};
    bucket_accumulate<algr::test_add_reducer>
        <<<1, 1>>>(bucket_sums.data(), algb::identity_mapper{generators.data()}, scalars.data(), 1);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      REQUIRE(bucket_sums[i] == 0);
    }
  }
}
