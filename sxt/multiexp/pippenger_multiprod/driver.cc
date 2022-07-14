#include "sxt/multiexp/pippenger_multiprod/driver.h"

#include "sxt/base/container/span_void.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_naive_multiproduct
//--------------------------------------------------------------------------------------------------
void driver::compute_naive_multiproduct(
    basct::span_void inout, basct::span<basct::span<uint64_t>> products,
    size_t num_inactive_inputs) const noexcept {
  this->compute_naive_multiproduct(
      inout,
      {reinterpret_cast<basct::cspan<uint64_t>*>(products.data()),
       products.size()},
      num_inactive_inputs);
}
} // namespace sxt::mtxpmp
