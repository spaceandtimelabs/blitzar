#pragma once

#include <cstddef>

#include "sxt/base/error/assert.h"
#include "sxt/base/container/span.h"
#include "sxt/execution/async/future.h"

namespace sxt::basdv { class stream; }

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// copy_host_to_device 
//--------------------------------------------------------------------------------------------------
xena::future<> copy_host_to_device(std::byte* dst, const basdv::stream& stream,
                                           const std::byte* src, size_t n, size_t count,
                                           size_t stride) noexcept;

template <class T>
xena::future<> copy_host_to_device(basct::span<T> dst, const basdv::stream& stream,
                                           basct::cspan<T> src, size_t stride, size_t slice_size,
                                           size_t offset) noexcept {
  if (slice_size == 0) {
    SXT_RELEASE_ASSERT(dst.empty());
    return xena::make_ready_future();
  }
  auto count = dst.size() / slice_size;
  SXT_RELEASE_ASSERT(
      // clang-format off
      stride >= slice_size &&
      dst.size() == count * slice_size && 
      src.size() >= offset + (count - 1u)*stride + slice_size
      // clang-format on
  );
  return copy_host_to_device(reinterpret_cast<std::byte*>(dst.data()), stream,
                                     reinterpret_cast<const std::byte*>(src.data() + offset),
                                     slice_size * sizeof(T), count, stride * sizeof(T));
}
} // namespace sxt::xendv
