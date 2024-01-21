#include "sxt/multiexp/bucket_method/reduction.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// plan_reduction 
//--------------------------------------------------------------------------------------------------
unsigned plan_reduction(unsigned num_buckets, unsigned num_outputs) noexcept {
  (void)num_buckets;
  (void)num_outputs;
  return 1;
}
} // namespace sxt::mtxbk
