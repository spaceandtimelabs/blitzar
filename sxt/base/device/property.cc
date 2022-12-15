#include "sxt/base/device/property.h"

#include <cuda_runtime.h>

#include <iostream>

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// get_num_devices
//--------------------------------------------------------------------------------------------------
int get_num_devices() noexcept {
  int num_devices;

  auto rcode = cudaGetDeviceCount(&num_devices);

  if (rcode != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(rcode) << "\n";

    return 0;
  }

  return num_devices;
}
} // namespace sxt::basdv
