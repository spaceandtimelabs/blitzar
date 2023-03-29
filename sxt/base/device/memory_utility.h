#pragma once

#include <cstddef>
#include <memory>

#include "sxt/base/concept/memcpyable_ranges.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept;

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
// async_memset_device
//--------------------------------------------------------------------------------------------------
void async_memset_device(void* dst, int val, size_t count, bast::raw_stream_t stream) noexcept;

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
// is_device_pointer
//--------------------------------------------------------------------------------------------------
bool is_device_pointer(const void* ptr) noexcept;

//--------------------------------------------------------------------------------------------------
// is_host_pointer
//--------------------------------------------------------------------------------------------------
bool is_host_pointer(const void* ptr) noexcept;
} // namespace sxt::basdv
