#include "sxt/multiexp/base/scalar_array.h"

#include "sxt/base/device/memory_utility.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// make_device_scalar_array 
//--------------------------------------------------------------------------------------------------
void make_device_scalar_array(basct::span<uint8_t> array, const basdv::stream& stream,
                              basct::cspan<uint8_t*> scalars, size_t element_num_bytes,
                              size_t n) noexcept {
  (void)array;
  (void)stream;
  (void)scalars;
  (void)element_num_bytes;
  (void)n;
}
} // namespace sxt::mtxb
