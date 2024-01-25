#include "sxt/multiexp/bucket_method/sum.h"

#include "sxt/algorithm/iteration/kernel_fit.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// fit_bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
void fit_bucket_sum_kernel(unsigned& num_blocks, unsigned& num_threads,
                           unsigned num_buckets) noexcept {
  auto dims = algi::fit_iteration_kernel(num_buckets);
  num_blocks = dims.num_blocks;
  num_threads = static_cast<unsigned>(dims.block_size);
}
} // namespace sxt::mtxbk
