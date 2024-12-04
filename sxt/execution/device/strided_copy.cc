#include "sxt/execution/device/strided_copy.h"

#include <cassert>
#include <cstring>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/pinned_buffer.h"
#include "sxt/base/device/stream.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy 
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy(std::byte* dst, const basdv::stream& stream, const std::byte* src,
                            size_t n, size_t count, size_t stride) noexcept {
  auto num_bytes = n * count;
  basdv::pinned_buffer buffer;
  assert(num_bytes <= buffer.size() && "todo");
  auto data = static_cast<std::byte*>(buffer.data());
  for (size_t i=0; i<count; ++i) {
    std::memcpy(data, src, n);
    data += n;
    src += stride;
  }
  basdv::async_memcpy_host_to_device(static_cast<void*>(dst), buffer.data(), num_bytes, stream);
  co_await await_stream(stream);
}
} // namespace sxt::xendv
