#include "benchmark/multi_exp1/multi_exp_gpu.h"

#include "benchmark/multi_exp1/multiply_add.h"
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
  auto tid = threadIdx.x;
  auto& reduction = reductions[tid];
  reduction = c21cn::zero_p3_v;
  for (int i=first; i<last; i+=num_threads_v) {
    multiply_add(reduction, mi, i);
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
// multi_exp_kernel 
//--------------------------------------------------------------------------------------------------
__global__ static void multi_exp_kernel(c21t::element_p3* res, int n) {
  __shared__ c21t::element_p3 reductions[num_threads_v];
  auto first = threadIdx.x;
  int mi = blockIdx.x;
  compute_reduction(res[mi], reductions, mi, first, n);
}

//--------------------------------------------------------------------------------------------------
// multi_exp_gpu
//--------------------------------------------------------------------------------------------------
void multi_exp_gpu(c21t::element_p3* res, int m, int n) noexcept {
  c21t::element_p3* device_elements;
  cudaMalloc(&device_elements, m * sizeof(c21t::element_p3));

  multi_exp_kernel<<<m, num_threads_v>>>(device_elements, n);

  cudaMemcpy(res, device_elements, m * sizeof(c21t::element_p3),
             cudaMemcpyDeviceToHost);
  cudaFree(device_elements);
}
} // namespace sxt

