#pragma once

#include "sxt/execution/async/future_state.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// leaf_continuation
//--------------------------------------------------------------------------------------------------
/**
 * A continuation that discards the result when its associated promise becomes available.
 */
template <class T> class leaf_continuation final : public task {
public:
  explicit leaf_continuation(future_state<T>&& state) noexcept : state_{std::move(state)} {}

  future_state<T>& state() noexcept { return state_; }

  // task
  void run_and_dispose() noexcept override { delete this; }

private:
  future_state<T> state_;
};
} // namespace sxt::xena
