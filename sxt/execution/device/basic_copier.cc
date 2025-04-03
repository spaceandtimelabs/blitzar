#include "sxt/execution/device/basic_copier.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
basic_copier::basic_copier(basct::span<std::byte> dst, basdv::stream& stream) noexcept
    : dst_{dst}, stream_{stream} {}

//--------------------------------------------------------------------------------------------------
// copy
//--------------------------------------------------------------------------------------------------
xena::future<> basic_copier::copy(basct::cspan<std::byte> src) noexcept {
  SXT_RELEASE_ASSERT(src.size() <= dst_.size());
  if (dst_.empty() || src.empty()) {
    co_return;
  }
  while (true) {
    if (src.empty()) {
      co_return;
    }
    SXT_RELEASE_ASSERT(!active_buffer_.empty());
    src = active_buffer_.fill_from_host(src);
    if (active_buffer_.size() == dst_.size()) {
      break;
    }
    if (!active_buffer_.full()) {
      SXT_DEBUG_ASSERT(src.empty());
      co_return;
    }
    if (!alt_buffer_.empty()) {
      co_await await_stream(stream_);
      alt_buffer_.reset();
    }
    basdv::async_memcpy_host_to_device(static_cast<void*>(dst_.data()), active_buffer_.data(),
                                       active_buffer_.size(), stream_);
    dst_ = dst_.subspan(active_buffer_.size());
    std::swap(active_buffer_, alt_buffer_);
  }

    // fill active buffer
   // assert(!active_buffer_.empty());
   // copy(active_buffer_, src);
   // src <- rest


  // if (!active_buffer.full() && !dst.full()) {
  //    co_return;
  // }
  //
  // fut = copy(dst, active_buffer);
  // co_await alt_future;
  // if (dst.full()) {
  //    co_await fut;
  //    co_return;
  // }
  // std::swap(active_buffer_, alt_buffer_);
  // alt_future = std::move(fut);
}
} // sxt::xendv
