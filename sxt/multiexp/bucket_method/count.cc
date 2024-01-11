#include "sxt/multiexp/bucket_method/count.h"

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_bucket_entries
//--------------------------------------------------------------------------------------------------
xena::future<> count_bucket_entries(basct::span<unsigned> count_array,
                                    basct::cspan<uint8_t> scalars, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_partitions) noexcept {
  (void)count_array;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)num_partitions;
  return {};
}
} // namespace sxt::mtxbk
