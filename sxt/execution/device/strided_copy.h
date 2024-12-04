#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future.h"

namespace sxt::basdv { class stream; }

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy_host_to_device 
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy_host_to_device(std::byte* dst, const basdv::stream& stream,
                                           const std::byte* src, size_t n, size_t count,
                                           size_t stride) noexcept;

template <class T>
xena::future<> strided_copy_host_to_device(basct::span<T> dst, const basdv::stream& stream,
                                           basct::cspan<T> src, size_t stride, size_t slice_size,
                                           size_t offset) noexcept {
  auto count = dst.size() / slice_size;
  return strided_copy_host_to_device(reinterpret_cast<std::byte*>(dst.data()), stream,
                                     reinterpret_cast<const std::byte*>(src.data() + offset),
                                     slice_size * sizeof(T), count, stride);
}
} // namespace sxt::xendv
