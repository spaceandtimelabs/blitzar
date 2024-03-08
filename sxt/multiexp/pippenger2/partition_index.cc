#include "sxt/multiexp/pippenger2/partition_index.h"

#include "sxt/execution/async/future.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// fill_partition_indexes 
//--------------------------------------------------------------------------------------------------
xena::future<> fill_partition_indexes(basct::span<uint16_t> indexes, basct::cspan<uint8_t*> scalars,
                                      unsigned element_num_bytes) noexcept {
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  return {};
}
} // namespace sxt::mtxpp2
