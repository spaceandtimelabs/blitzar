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

#include <iostream>
#include <string>

#include "sxt/base/device/pointer_attributes.h"
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
    baser::panic("cudaMemcpy failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void memcpy_device_to_host(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpy failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_device(void* dst, const void* src, size_t count,
                                   bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_host(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_peer
//--------------------------------------------------------------------------------------------------
void async_memcpy_peer(void* dst, int dst_device, const void* src, int src_device, size_t count,
                       bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyPeerAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_to_device(void* dst, const void* src, size_t count,
                            const pointer_attributes& attrs, const stream& stream) noexcept {
  SXT_DEBUG_ASSERT(attrs.device != stream.device());
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
    baser::panic("cudaMemsetAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// memset_device
//--------------------------------------------------------------------------------------------------
void memset_device(void* dst, int value, size_t count) noexcept {
  auto rcode = cudaMemset(dst, value, count);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemset failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// get_pointer_attributes
//--------------------------------------------------------------------------------------------------
void get_pointer_attributes(pointer_attributes& attrs, const void* ptr) noexcept {
  cudaPointerAttributes cuda_attrs;
  auto rcode = cudaPointerGetAttributes(&cuda_attrs, ptr);
  if (rcode != cudaSuccess) {
    baser::panic("cudaPointerGetAttributes failed: " + std::string(cudaGetErrorString(rcode)));
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
    baser::panic("cudaPointerGetAttributes failed: " + std::string(cudaGetErrorString(rcode)));
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
    baser::panic("cudaPointerGetAttributes failed: " + std::string(cudaGetErrorString(rcode)));
  }
  return attrs.type != cudaMemoryTypeDevice;
}
} // namespace sxt::basdv
