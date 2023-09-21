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
  memmg::managed_array<uint64_t> scalars(memr::get_managed_device_resource());
}
