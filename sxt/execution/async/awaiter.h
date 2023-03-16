#pragma once

#include <type_traits>

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine_promise.h"
#include "sxt/execution/async/future.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// awaiter
//--------------------------------------------------------------------------------------------------
template <class T> class awaiter {
public:
  explicit awaiter(future<T>&& fut) noexcept : future_{std::move(fut)} {}

  bool await_ready() const noexcept { return future_.ready(); }

  template <class U>
  void await_suspend(std::coroutine_handle<coroutine_promise<U>> handle) noexcept {
    SXT_DEBUG_ASSERT(!future_.ready(), "we don't support preemptive futures");
    auto pr = future_.promise();
    SXT_DEBUG_ASSERT(pr != nullptr, "future must have an attached promise");
    pr->set_continuation(handle.promise());
  }

  T await_resume() noexcept {
    if constexpr (std::is_same_v<T, void>) {
      return;
    } else {
      return T{std::move(future_.value())};
    }
  }

private:
  future<T> future_;
};

extern template class awaiter<void>;
} // namespace sxt::xena
