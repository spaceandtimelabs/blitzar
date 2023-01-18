#include "sxt/base/device/synchronization.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// synchronize_stream
//--------------------------------------------------------------------------------------------------
void synchronize_stream(CUstream_st* stream) noexcept {
  auto rcode = cudaStreamSynchronize(stream);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(rcode) << "\n";
    std::abort();
  }
}
} // namespace sxt::basdv
