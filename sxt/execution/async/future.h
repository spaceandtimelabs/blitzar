#pragma once

#include <utility>

#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/future_completion_fn.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/execution/async/future_ready.h"
#include "sxt/execution/async/future_value_storage.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future
//--------------------------------------------------------------------------------------------------
/**
 * Manage asynchronous computations.
 *
 * This is a highly simplified version of a future. It doesn't support chaining, multiplexing, or
 * error results. It's expected that we'll rewrite this class, but it can serve as a placeholder to
 * get basic algorithms out.
 *
 * See seastar's futures (https://seastar.io/futures-promises/) for a model of what we'd like to
 * build towards.
 */
template <class T> class future {
public:
  using completion_fn = future_completion_fn<T>;

  future() noexcept = default;

  explicit future(future_ready_tag /*tag*/) noexcept : is_available_{true} {}

  template <class ValRRef = std::add_rvalue_reference_t<T>>
  future(ValRRef value, future_ready_tag /*tag*/) noexcept
      : is_available_{true}, value_{std::move(value)} {}

  explicit future(computation_handle&& computation,
                  completion_fn&& on_computation_done = {}) noexcept
      : on_computation_done_{std::move(on_computation_done)}, computation_{std::move(computation)} {
  }

  template <class ValRRef = std::add_rvalue_reference_t<T>>
  future(ValRRef value, computation_handle&& computation,
         completion_fn&& on_computation_done = {}) noexcept
      : value_{std::move(value)}, on_computation_done_{std::move(on_computation_done)},
        computation_{std::move(computation)} {}

  future(const future&) = delete;
  future(future&&) noexcept = default;

  future& operator=(const future&) = delete;
  future& operator=(future&& other) noexcept {
    // Note: computation_ is move assigned before the on_computation_done_ functor because
    // computation_ may depend on resources owned by on_computation_done_, and move assignment
    // may block until it is completed
    computation_ = std::move(other.computation_);
    on_computation_done_ = std::move(other.on_computation_done_);
    value_ = std::move(other.value_);
    is_available_ = other.is_available_;

    return *this;
  }

  bool available() const noexcept { return is_available_; }

  void wait() noexcept {
    if (is_available_) {
      return;
    }
    computation_.wait();
    if (on_computation_done_) {
      if constexpr (std::is_same_v<T, void>) {
        on_computation_done_();
      } else {
        on_computation_done_(value_);
      }
      on_computation_done_ = completion_fn{};
    }
    is_available_ = true;
  }

  T await_result() noexcept {
    this->wait();
    return value_.consume_value();
  }

private:
  bool is_available_{false};
  [[no_unique_address]] future_value_storage<T> value_;
  completion_fn on_computation_done_;
  computation_handle computation_;
};

//--------------------------------------------------------------------------------------------------
// make_ready_future
//--------------------------------------------------------------------------------------------------
template <class T> future<T> make_ready_future(T&& value) noexcept {
  return future<T>{std::move(value), future_ready_v};
}

inline future<void> make_ready_future() noexcept { return future<void>{future_ready_v}; }
} // namespace sxt::xena
