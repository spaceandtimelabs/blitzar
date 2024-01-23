#include "sxt/multiexp/bucket_method/multiexponentiation2.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// plan_multiexponentiation 
//--------------------------------------------------------------------------------------------------
void plan_multiexponentiation(multiexponentiate_options& options, unsigned num_outputs,
                              unsigned element_num_bytes, unsigned n) noexcept {
  (void)options;
  (void)num_outputs;
  (void)element_num_bytes;
  (void)n;
  options = multiexponentiate_options{};
}
} // namespace sxt::mtxbk
