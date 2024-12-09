#include "sxt/execution/device/strided_copy.h"

#include <cassert>
#include <cstring>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/pinned_buffer.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy_host_to_device_one_sweep 
//--------------------------------------------------------------------------------------------------
static xena::future<> strided_copy_host_to_device_one_sweep(std::byte* dst,
                                                            const basdv::stream& stream,
                                                            const std::byte* src, size_t n,
                                                            size_t count, size_t stride) noexcept {
  auto num_bytes = n * count;
  if (num_bytes == 0) {
    co_return;
  }
  basdv::pinned_buffer buffer;
  auto data = static_cast<std::byte*>(buffer.data());
  for (size_t i=0; i<count; ++i) {
    std::memcpy(data, src, n);
    data += n;
    src += stride;
  }
  basdv::async_memcpy_host_to_device(static_cast<void*>(dst), buffer.data(), num_bytes, stream);
  co_await await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// strided_copy_host_to_device 
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy_host_to_device(std::byte* dst, const basdv::stream& stream,
                                           const std::byte* src, size_t n, size_t count,
                                           size_t stride) noexcept {
  auto num_bytes = n * count;
  if (num_bytes <= basdv::pinned_buffer::size()) {
    co_return co_await strided_copy_host_to_device_one_sweep(dst, stream, src, n, count, stride);
  }
  auto cur_n = n;
   
  auto fill_buffer = [&] (basdv::pinned_buffer& buffer) noexcept {
    size_t remaining_size = buffer.size();
    auto data = static_cast<std::byte*>(buffer.data());
    while (remaining_size > 0 && count > 0) {
      auto chunk_size = std::min(remaining_size, cur_n);
      std::memcpy(data, src, chunk_size);
      src += chunk_size;
      data += chunk_size;
      remaining_size -= chunk_size;
      cur_n -= chunk_size;
      if (cur_n == 0) {
        --count;
        cur_n = n;
        src += stride - n;
      }
    }
    return buffer.size() - remaining_size;
  };
  
  // copy
  basdv::pinned_buffer cur_buffer, alt_buffer;
  auto chunk_size = fill_buffer(cur_buffer);
  SXT_DEBUG_ASSERT(count > 0, "copy can't be done in a single sweep");
  while (count > 0) {
    basdv::async_memcpy_host_to_device(static_cast<void*>(dst), cur_buffer.data(), chunk_size, stream);
    dst += chunk_size;
    chunk_size = fill_buffer(alt_buffer);
    co_await await_stream(stream);
    std::swap(cur_buffer, alt_buffer);
  }
  basdv::async_memcpy_host_to_device(static_cast<void*>(dst), cur_buffer.data(), chunk_size, stream);
  co_await await_stream(stream);
}
} // namespace sxt::xendv
