#include "sxt/base/device/synchronization.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// synchronize_stream
//--------------------------------------------------------------------------------------------------
void synchronize_stream(bast::raw_stream_t stream) noexcept {
  auto rcode = cudaStreamSynchronize(stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(rcode)));
  }
}
} // namespace sxt::basdv
