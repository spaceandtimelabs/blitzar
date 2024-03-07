#include "sxt/multiexp/pippenger2/partition_index_kernel.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// scalar32
//--------------------------------------------------------------------------------------------------
namespace {
struct scalar32 {
  uint8_t data[32];
};
} // namespace

//--------------------------------------------------------------------------------------------------
// fill_partition_index_kernel_impl
//--------------------------------------------------------------------------------------------------
__device__ void fill_partition_index_kernel_impl(uint16_t* __restrict__ indexes,
                                                 const scalar32* __restrict__ scalars,
                                                 unsigned num_outputs, unsigned n) {
  auto thread_index = threadIdx.x;
  auto output_index = blockIdx.x;
  auto byte_index = thread_index / 8u;
  auto bit_index = thread_index % 8u;
  uint8_t mask = static_cast<uint8_t>(1u << bit_index);

  // adjust pointers
  indexes += output_index * 256u;
  scalars += output_index * n;

  // fill indexes
  __shared__ scalar32 buffer[16];
  for (unsigned i = 0; i < n; i += 16) {
    // load scalars into shared memory
    if (thread_index < 16) {
      if (i + thread_index < n) {
        buffer[thread_index] = scalars[i + thread_index];
      } else {
        buffer[thread_index] = {};
      }
    }
    __syncthreads();

    // compute index
    uint16_t index = 0;
    for (uint16_t j=0; j<16; ++j) {
      index += static_cast<uint16_t>((buffer[j].data[byte_index] & mask) != 0) << j;
    }

    // write results
    indexes[thread_index] = index;
    indexes += num_outputs * 256u;
  }
}

//--------------------------------------------------------------------------------------------------
// fill_partition_index_kernel 
//--------------------------------------------------------------------------------------------------
__global__ void fill_partition_index_kernel(uint16_t* __restrict__ indexes,
                                            const uint8_t* __restrict__ scalars,
                                            unsigned num_outputs, unsigned n) {
  fill_partition_index_kernel_impl(indexes, reinterpret_cast<const scalar32*>(scalars), num_outputs,
                                   n);
}
} // namespace sxt::mtxpp2
