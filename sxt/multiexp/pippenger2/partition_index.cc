#include "sxt/multiexp/pippenger2/partition_index.h"

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// fill_partition_indexes 
//--------------------------------------------------------------------------------------------------
xena::future<> fill_partition_indexes(basct::span<uint16_t> indexes, basct::cspan<uint8_t*> scalars,
                                      unsigned element_num_bytes, unsigned n) noexcept {
  auto num_indexes_per_product = basn::divide_up(n, 16u);
  auto num_outputs = scalars.size();
  auto num_indexes = num_indexes_per_product * num_outputs * element_num_bytes * 8u;
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(indexes.data()) &&
      indexes.size() == num_indexes
      // clang-format on
  );
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<uint8_t> scalar_array{n * num_outputs * element_num_bytes, &resource};
  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    basdv::async_copy_host_to_device(
        basct::subspan(scalar_array, output_index * n * element_num_bytes, n * element_num_bytes),
        basct::span<uint8_t>{scalars[output_index], n * element_num_bytes}, stream);
  }
  (void)scalar_array;
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  return {};
}
} // namespace sxt::mtxpp2
