#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// make_multiproduct_table 
//--------------------------------------------------------------------------------------------------
xena::future<> make_multiproduct_table(basct::span<unsigned> bucket_prefix_counts,
                                       memmg::managed_array<unsigned>& indexes,
                                       basct::cspan<const uint8_t*> scalars,
                                       unsigned element_num_bytes, unsigned bit_width,
                                       unsigned n) noexcept {
  (void)bucket_prefix_counts;
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)n;
  return {};
}
} // namespace sxt::mtxb
