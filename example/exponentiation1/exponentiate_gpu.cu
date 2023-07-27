#include "example/exponentiation1/exponentiate_gpu.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/operation/scalar_multiply.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// kernel
//--------------------------------------------------------------------------------------------------
static __global__ void kernel(c21t::element_p3* res, int n) {
  c21t::element_p3 g{
      {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      {1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      {1, 0, 0, 0, 0},
      {1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    /* basn::fast_random_number_generator generator{i+1, i+2}; */
    unsigned char a[32] = {1};
    /* e21c::generate_random_exponent(a, generator); */
    c21o::scalar_multiply(res[i], a, g);
    /* res[i] = g; */
  }
}

//--------------------------------------------------------------------------------------------------
// exponentiate_gpu
//--------------------------------------------------------------------------------------------------
void exponentiate_gpu(c21t::element_p3* res, int n) noexcept {
  c21t::element_p3* res_p;
  cudaMalloc(&res_p, n * sizeof(c21t::element_p3));
  kernel<<<1, 128>>>(res_p, n);
  cudaMemcpy(res, res_p, n * sizeof(c21t::element_p3), cudaMemcpyDeviceToHost);
  cudaFree(res_p);
}
}  // namespace sxt
