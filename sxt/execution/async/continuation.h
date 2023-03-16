#pragma once

#include <type_traits>

#include "sxt/execution/async/continuation_fn.h"
#include "sxt/execution/async/future_state.h"
#include "sxt/execution/async/future_state_utility.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// continuation
//--------------------------------------------------------------------------------------------------
template <class T, class Tp, continuation_fn<T, Tp> F> class continuation final : public task {
public:
  continuation(future_state<T>&& state, F&& f) noexcept
      : state_{std::move(state)}, f_{std::move(f)} {}

  continuation(const continuation&) = delete;
  continuation(continuation&&) noexcept = delete;
  continuation& operator=(const continuation&) = delete;
  continuation& operator=(continuation&&) noexcept = delete;

  future_state<T>& state() noexcept { return state_; }

  promise<Tp>& promise_p() noexcept { return promise_p_; }

  // task
  void run_and_dispose() noexcept override {
    invoke_continuation_fn(promise_p_.state(), f_, state_);
    promise_p_.make_ready();
    delete this;
  }

private:
  future_state<T> state_;
  promise<Tp> promise_p_;
  F f_;
};
} // namespace sxt::xena
