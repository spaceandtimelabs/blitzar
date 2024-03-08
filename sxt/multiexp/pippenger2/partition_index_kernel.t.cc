#include "sxt/multiexp/pippenger2/partition_index_kernel.h"

#include <vector>

#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the partition indexes for a multiexponentiation") {
  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::pmr::vector<uint8_t> scalars{memr::get_managed_device_resource()};

  indexes.resize(256);
  scalars.resize(32);

  SECTION("we handle a single scalar of zero") {
    /* fill_partition_indexes_kernel<<<1, 256>>>(indexes.data(), scalars.data(), 1, 1); */
  }
  /* __global__ void fill_partition_index_kernel(uint16_t* __restrict__ indexes, */
  /*                                             const uint8_t* __restrict__ scalars, */
  /*                                             unsigned num_outputs, unsigned n); */
}
