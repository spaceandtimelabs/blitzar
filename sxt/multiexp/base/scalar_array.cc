#include "sxt/multiexp/base/scalar_array.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// make_device_scalar_array 
//--------------------------------------------------------------------------------------------------
void make_device_scalar_array(basct::span<uint8_t> array, const basdv::stream& stream,
                              basct::cspan<uint8_t*> scalars, size_t element_num_bytes,
                              size_t n) noexcept {
  auto num_outputs = scalars.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      array.size() == num_outputs * element_num_bytes * n &&
      basdv::is_active_device_pointer(array.data())
      // clang-format on
  );
  auto num_bytes_per_output = element_num_bytes * n;
  for (size_t output_index = 0; output_index<num_outputs; ++output_index) {
    auto output_first = output_index * element_num_bytes * n;
    SXT_DEBUG_ASSERT(basdv::is_host_pointer(scalars[output_index]));
    basdv::async_copy_host_to_device(
        array.subspan(output_first, num_bytes_per_output),
        basct::cspan<uint8_t>{scalars[output_index], num_bytes_per_output}, stream);
  }
}
} // namespace sxt::mtxb
