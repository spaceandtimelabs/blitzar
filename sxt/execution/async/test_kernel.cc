#include "sxt/execution/async/test_kernel.h"

#include "sxt/base/num/divide_up.h"
#include "sxt/execution/base/stream.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// add_impl
//--------------------------------------------------------------------------------------------------
static __global__ void add_impl(uint64_t* c, uint64_t* a, uint64_t* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

//--------------------------------------------------------------------------------------------------
// add_for_testing
//--------------------------------------------------------------------------------------------------
void add_for_testing(uint64_t* c, const xenb::stream& stream, uint64_t* a, uint64_t* b,
                     int n) noexcept {
  add_impl<<<basn::divide_up(n, 256), 256, 0, stream.raw_stream()>>>(c, a, b, n);
}
} // namespace sxt::xena
