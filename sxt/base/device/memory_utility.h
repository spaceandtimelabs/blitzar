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
#pragma once

#include <cstddef>
#include <memory>

#include "sxt/base/concept/memcpyable_ranges.h"
#include "sxt/base/container/span.h"
#include "sxt/base/device/pointer_attributes.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
class stream;
struct pointer_attributes;

//--------------------------------------------------------------------------------------------------
// memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept;

//--------------------------------------------------------------------------------------------------
// memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void memcpy_device_to_host(void* dst, const void* src, size_t count) noexcept;

//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_device(void* dst, const void* src, size_t count,
                                   bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_host(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_memcpy_peer
//--------------------------------------------------------------------------------------------------
void async_memcpy_peer(void* dst, int dst_device, const void* src, int src_device, size_t count,
                       bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_memset_device
//--------------------------------------------------------------------------------------------------
void async_memset_device(void* dst, int val, size_t count, bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_memcpy_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_to_device(void* dst, const void* src, size_t count,
                            const pointer_attributes& attrs, const stream& stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_copy_host_to_device
//--------------------------------------------------------------------------------------------------
template <class Dst, class Src>
  requires bascpt::memcpyable_ranges<Dst, Src>
void async_copy_host_to_device(Dst&& dst, const Src& src, bast::raw_stream_t stream) noexcept {
  SXT_DEBUG_ASSERT(dst.size() == src.size());
  auto dst_data = std::to_address(std::begin(dst));
  using T = std::remove_cvref_t<decltype(*dst_data)>;
  async_memcpy_host_to_device(dst_data, std::to_address(std::begin(src)), sizeof(T) * dst.size(),
                              stream);
}

//--------------------------------------------------------------------------------------------------
// async_copy_to_device
//--------------------------------------------------------------------------------------------------
template <class Dst, class Src>
  requires bascpt::memcpyable_ranges<Dst, Src>
void async_copy_to_device(Dst&& dst, const Src& src, const stream& stream) noexcept {
  SXT_DEBUG_ASSERT(dst.size() == src.size());
  auto dst_data = std::to_address(std::begin(dst));
  using T = std::remove_cvref_t<decltype(*dst_data)>;
  pointer_attributes src_attrs;
  auto src_ptr = src.data();
  get_pointer_attributes(src_attrs, src_ptr);
  async_memcpy_to_device(dst.data(), src_ptr, sizeof(T) * dst.size(), src_attrs, stream);
}

//--------------------------------------------------------------------------------------------------
// async_copy_device_to_host
//--------------------------------------------------------------------------------------------------
template <class Dst, class Src>
  requires bascpt::memcpyable_ranges<Dst, Src>
void async_copy_device_to_host(Dst&& dst, const Src& src, bast::raw_stream_t stream) noexcept {
  SXT_DEBUG_ASSERT(dst.size() == src.size());
  auto dst_data = std::to_address(std::begin(dst));
  using T = std::remove_cvref_t<decltype(*dst_data)>;
  async_memcpy_device_to_host(dst_data, std::to_address(std::begin(src)), sizeof(T) * dst.size(),
                              stream);
}

//--------------------------------------------------------------------------------------------------
// async_copy_device_to_device
//--------------------------------------------------------------------------------------------------
template <class Dst, class Src>
  requires bascpt::memcpyable_ranges<Dst, Src>
void async_copy_device_to_device(Dst&& dst, const Src& src, bast::raw_stream_t stream) noexcept {
  SXT_DEBUG_ASSERT(dst.size() == src.size());
  auto dst_data = std::to_address(std::begin(dst));
  using T = std::remove_cvref_t<decltype(*dst_data)>;
  async_memcpy_device_to_device(dst_data, std::to_address(std::begin(src)), sizeof(T) * dst.size(),
                                stream);
}

//--------------------------------------------------------------------------------------------------
// memset_device
//--------------------------------------------------------------------------------------------------
void memset_device(void* dst, int value, size_t count) noexcept;

//--------------------------------------------------------------------------------------------------
// get_pointer_attributes
//--------------------------------------------------------------------------------------------------
void get_pointer_attributes(pointer_attributes& attrs, const void* ptr) noexcept;

//--------------------------------------------------------------------------------------------------
// is_active_device_pointer
//--------------------------------------------------------------------------------------------------
bool is_active_device_pointer(const void* ptr) noexcept;

//--------------------------------------------------------------------------------------------------
// is_host_pointer
//--------------------------------------------------------------------------------------------------
bool is_host_pointer(const void* ptr) noexcept;

//--------------------------------------------------------------------------------------------------
// is_equal_for_testing
//--------------------------------------------------------------------------------------------------
bool is_equal_for_testing(const void* lhs, const void* rhs, size_t size) noexcept;

template <class T> bool is_equal_for_testing(basct::cspan<T> lhs, basct::cspan<T> rhs) noexcept {
  return lhs.size() == rhs.size() &&
         is_equal_for_testing(lhs.data(), rhs.data(), sizeof(T) * lhs.size());
}

//--------------------------------------------------------------------------------------------------
// get_mem_info
//--------------------------------------------------------------------------------------------------
void get_mem_info(size_t& bytes_free, size_t& bytes_total) noexcept;
} // namespace sxt::basdv
