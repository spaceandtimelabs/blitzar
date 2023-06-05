/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/base/device/property.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// get_num_devices
//--------------------------------------------------------------------------------------------------
int get_num_devices() noexcept {
  static int num_devices = []() noexcept {
    int res;
    auto rcode = cudaGetDeviceCount(&res);
    if (rcode != cudaSuccess) {
      std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(rcode) << "\n";
      return 0;
    }
    return res;
  }();
  return num_devices;
}

//--------------------------------------------------------------------------------------------------
// get_stream_device
//--------------------------------------------------------------------------------------------------
int get_stream_device(bast::raw_stream_t stream) noexcept {
  CUcontext ctx;
  auto rcode = cuStreamGetCtx(stream, &ctx);
  if (rcode != 0) {
    baser::panic("cuStreamGetCtx failed\n");
  }
  rcode = cuCtxPushCurrent(ctx);
  if (rcode != 0) {
    baser::panic("cuCtxPushCurrent failed\n");
  }
  CUdevice device;
  rcode = cuCtxGetDevice(&device);
  if (rcode != 0) {
    baser::panic("cuCtxGetDevice failed\n");
  }
  rcode = cuCtxPopCurrent(&ctx);
  if (rcode != 0) {
    baser::panic("cuCtxPopCurrent failed\n");
  }
  return device;
}
} // namespace sxt::basdv
