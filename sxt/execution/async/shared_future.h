#pragma once

#include "sxt/execution/async/future.h"
#include "sxt/execution/async/shared_future_state.h"
#include <utility>

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// shared_future
//--------------------------------------------------------------------------------------------------
template <class T=void> class shared_future {
public:
  shared_future() noexcept = default;

  shared_future(future<T>&& fut) noexcept
      : state_{std::make_shared<shared_future_state<T>>(std::move(fut))} {}

  future<T> get_future() const noexcept { return state_->get_future(); }

private:
  std::shared_ptr<shared_future_state<T>> state_;
};
} // namespace sxt::xena
