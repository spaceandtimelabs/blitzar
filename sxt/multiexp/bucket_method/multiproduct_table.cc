#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "sxt/memory/management/managed_array.h"
#include "sxt/execution/async/future.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_table
//--------------------------------------------------------------------------------------------------
xena::future<void> compute_multiproduct_table(memmg::managed_array<bucket_descriptor>& table,
                                              memmg::managed_array<unsigned>& indexes,
                                              basct::cspan<uint8_t> scalars,
                                              unsigned element_num_bytes,
                                              unsigned bit_width) noexcept {
  (void)table;
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  return {};
}
} // namespace sxt::mtxbk
