#include "sxt/base/device/event_utility.h"

#include <cuda_runtime.h>

#include <string>

#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// record_event
//--------------------------------------------------------------------------------------------------
void record_event(bast::raw_cuda_event_t event, bast::raw_stream_t stream) noexcept {
  auto rcode = cudaEventRecord(event, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaEventRecord failed: " + std::string(cudaGetErrorString(rcode)));
  }
}
} // namespace sxt::basdv
