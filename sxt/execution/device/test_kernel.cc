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
#include "sxt/execution/device/test_kernel.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// add_impl
//--------------------------------------------------------------------------------------------------
static __global__ void add_impl(uint64_t* c, const uint64_t* a, const uint64_t* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

//--------------------------------------------------------------------------------------------------
// add_for_testing
//--------------------------------------------------------------------------------------------------
void add_for_testing(uint64_t* c, bast::raw_stream_t stream, const uint64_t* a, const uint64_t* b,
                     int n) noexcept {
  memr::async_device_resource resource{stream};

  memmg::managed_array<uint64_t> a_dev{&resource};
  memmg::managed_array<uint64_t> b_dev{&resource};
  memmg::managed_array<uint64_t> c_dev{&resource};
  auto cp = c;
  if (!basdv::is_device_pointer(a)) {
    a_dev = memmg::managed_array<uint64_t>{static_cast<unsigned>(n), &resource};
    basdv::async_memcpy_host_to_device(a_dev.data(), a, n * sizeof(uint64_t), stream);
    a = a_dev.data();
  }
  if (!basdv::is_device_pointer(b)) {
    b_dev = memmg::managed_array<uint64_t>{static_cast<unsigned>(n), &resource};
    basdv::async_memcpy_host_to_device(b_dev.data(), b, n * sizeof(uint64_t), stream);
    b = b_dev.data();
  }
  if (!basdv::is_device_pointer(c)) {
    c_dev = memmg::managed_array<uint64_t>{static_cast<unsigned>(n), &resource};
    cp = c_dev.data();
  }
  add_impl<<<basn::divide_up(n, 256), 256, 0, stream>>>(cp, a, b, n);
  if (!basdv::is_device_pointer(c)) {
    basdv::async_memcpy_device_to_host(c, cp, n * sizeof(uint64_t), stream);
  }
}
} // namespace sxt::xendv
