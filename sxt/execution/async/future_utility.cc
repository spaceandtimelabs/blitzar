#include "sxt/execution/async/future_utility.h"

#include "sxt/execution/async/coroutine.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// await_all
//--------------------------------------------------------------------------------------------------
future<> await_all(std::vector<future<>> futs) noexcept {
  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}
} // namespace sxt::xena
