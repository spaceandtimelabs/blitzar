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
#include "sxt/base/device/memory_utility.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpy failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void memcpy_device_to_host(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpy failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_device(void* dst, const void* src, size_t count,
                                   bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_host(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_peer
//--------------------------------------------------------------------------------------------------
void async_memcpy_peer(void* dst, int dst_device, const void* src, int src_device, size_t count,
                       bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyPeerAsync failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_to_device(void* dst, const void* src, size_t count,
                            const pointer_attributes& attrs, const stream& stream) noexcept {
  switch (attrs.kind) {
  case pointer_kind_t::host:
    return async_memcpy_host_to_device(dst, src, count, stream);
  case pointer_kind_t::device:
    return async_memcpy_peer(dst, stream.device(), src, attrs.device, count, stream);
  case pointer_kind_t::managed:
    baser::panic("async_memcpy_to_device doesn't support managed pointers");
  }
}

//--------------------------------------------------------------------------------------------------
// async_memset_device
//--------------------------------------------------------------------------------------------------
void async_memset_device(void* dst, int val, size_t count, bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemsetAsync(dst, val, count, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemsetAsync failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// memset_device
//--------------------------------------------------------------------------------------------------
void memset_device(void* dst, int value, size_t count) noexcept {
  auto rcode = cudaMemset(dst, value, count);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemset failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// get_pointer_attributes
//--------------------------------------------------------------------------------------------------
void get_pointer_attributes(pointer_attributes& attrs, const void* ptr) noexcept {
  cudaPointerAttributes cuda_attrs;
  auto rcode = cudaPointerGetAttributes(&cuda_attrs, ptr);
  if (rcode != cudaSuccess) {
    baser::panic("cudaPointerGetAttributes failed: {}", cudaGetErrorString(rcode));
  }
  if (cuda_attrs.type == cudaMemoryTypeDevice) {
    attrs.kind = pointer_kind_t::device;
    attrs.device = cuda_attrs.device;
    return;
  }
  if (cuda_attrs.type == cudaMemoryTypeManaged) {
    attrs.kind = pointer_kind_t::managed;
    attrs.device = cuda_attrs.device;
    return;
  }
  attrs.kind = pointer_kind_t::host;
  attrs.device = -1;
}

//--------------------------------------------------------------------------------------------------
// is_active_device_pointer
//--------------------------------------------------------------------------------------------------
bool is_active_device_pointer(const void* ptr) noexcept {
  cudaPointerAttributes attrs;
  auto rcode = cudaPointerGetAttributes(&attrs, ptr);
  if (rcode != cudaSuccess) {
    baser::panic("cudaPointerGetAttributes failed: {}", cudaGetErrorString(rcode));
  }
  if (attrs.type == cudaMemoryTypeManaged) {
    return true;
  }
  return attrs.type == cudaMemoryTypeDevice && attrs.device == get_device();
}

//--------------------------------------------------------------------------------------------------
// is_host_pointer
//--------------------------------------------------------------------------------------------------
bool is_host_pointer(const void* ptr) noexcept {
  cudaPointerAttributes attrs;
  auto rcode = cudaPointerGetAttributes(&attrs, ptr);
  if (rcode != cudaSuccess) {
    baser::panic("cudaPointerGetAttributes failed: {}", cudaGetErrorString(rcode));
  }
  return attrs.type != cudaMemoryTypeDevice;
}

//--------------------------------------------------------------------------------------------------
// is_equal_for_testing
//--------------------------------------------------------------------------------------------------
bool is_equal_for_testing(const void* lhs, const void* rhs, size_t size) noexcept {
  if (size == 0) {
    return true;
  }
  pointer_attributes attrs;

  // lhs
  std::vector<std::byte> lhs_data;
  get_pointer_attributes(attrs, lhs);
  if (attrs.kind == pointer_kind_t::device) {
    active_device_guard active_guard{attrs.device};
    lhs_data.resize(size);
    memcpy_device_to_host(lhs_data.data(), lhs, size);
    lhs = lhs_data.data();
  }

  // rhs
  std::vector<std::byte> rhs_data;
  get_pointer_attributes(attrs, rhs);
  if (attrs.kind == pointer_kind_t::device) {
    active_device_guard active_guard{attrs.device};
    rhs_data.resize(size);
    memcpy_device_to_host(rhs_data.data(), rhs, size);
    rhs = rhs_data.data();
  }

  // comparison
  return std::memcmp(lhs, rhs, size) == 0;
}
} // namespace sxt::basdv
