#include "sxt/execution/device/basic_copier.h"

#include "sxt/execution/async/coroutine.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// copy
//--------------------------------------------------------------------------------------------------
xena::future<> basic_copier::copy(basct::cspan<std::byte> src) noexcept {
  if (dst_.empty() || src.empty()) {
    co_return;
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
