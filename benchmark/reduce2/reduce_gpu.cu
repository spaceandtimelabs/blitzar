#include "benchmark/reduce2/reduce_gpu.h"

#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt {
constexpr int num_threads_v = 128;

//--------------------------------------------------------------------------------------------------
// compute_reduction 
//--------------------------------------------------------------------------------------------------
__device__ static void compute_reduction(c21t::element_p3& res_mi,
                                         c21t::element_p3* reductions, int mi,
                                         int first, int last) {
  // pretend like g is a random element rather than fixed
  c21t::element_p3 g{
      {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      {1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      {1, 0, 0, 0, 0},
      {1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  auto tid = threadIdx.x;
  auto& reduction = reductions[tid];
  reduction = c21cn::zero_p3_v;
  for (int i=first; i<last; i+=num_threads_v) {
    c21o::add(reduction, reduction, g);
  }

  __syncthreads();
  for (int s=num_threads_v/2; s>0; s>>=1) {
    if (tid < s) {
      c21o::add(reduction, reduction, reductions[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    res_mi = reduction;
  }
}


//--------------------------------------------------------------------------------------------------
// reduce_kernel 
//--------------------------------------------------------------------------------------------------
__global__ static void reduce_kernel(c21t::element_p3* res, int n) {
  __shared__ c21t::element_p3 reductions[num_threads_v];
  auto first = threadIdx.x;
  int mi = blockIdx.x;
  compute_reduction(res[mi], reductions, mi, first, n);
}

//--------------------------------------------------------------------------------------------------
// reduce_gpu
//--------------------------------------------------------------------------------------------------
void reduce_gpu(c21t::element_p3* res, int m, int n) noexcept {
  c21t::element_p3* device_elements;
  cudaMalloc(&device_elements, m * sizeof(c21t::element_p3));

  reduce_kernel<<<m, num_threads_v>>>(device_elements, n);

  cudaMemcpy(res, device_elements, m * sizeof(c21t::element_p3),
             cudaMemcpyDeviceToHost);
  cudaFree(device_elements);
}
} // namespace sxt

