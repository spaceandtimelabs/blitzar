#include "example/field_arithmetic/reduce1.h"

#include "sxt/field51/base/reduce.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// reduce_kernel
//--------------------------------------------------------------------------------------------------
__global__ static void reduce_kernel(f51t::element* res, int m, int n) {
  int m_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (m_i >= m) {
    return;
  }
  res[m_i] = f51t::element{static_cast<uint64_t>(m_i), 0, 0, 0, 0};
  for (int i = 0; i < n; ++i) {
    auto h = res[m_i];
    f51o::mul(h, h, h);
    f51o::add(res[m_i], res[m_i], h);
    f51b::reduce(res[m_i].data(), res[m_i].data());
  }
}

//--------------------------------------------------------------------------------------------------
// reduce1
//--------------------------------------------------------------------------------------------------
void reduce1(f51t::element* elements, int m, int n) noexcept {
  f51t::element* device_elements;
  cudaMalloc(&device_elements, m * sizeof(f51t::element));
  auto num_blocks = (m + 31) / 32;
  reduce_kernel<<<num_blocks, 32>>>(device_elements, m, n);
  cudaMemcpy(elements, device_elements, m * sizeof(f51t::element),
             cudaMemcpyDeviceToHost);
  cudaFree(device_elements);
}
}  // namespace sxt
