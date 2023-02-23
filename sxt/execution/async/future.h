#pragma once

#include <utility>

#include "sxt/base/error/assert.h"
#include "sxt/base/functional/move_only_function.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future
//--------------------------------------------------------------------------------------------------
/**
 * Manage asynchronous computations.
 *
 * This is a highly simplified version of a future. It doesn't support multiplexing or
 * error results. It's expected that we'll rewrite this class, but it can serve as a placeholder to
 * get basic algorithms out.
 *
 * See seastar's futures (https://seastar.io/futures-promises/) for a model of what we'd like to
 * build towards.
 */
template <class T> class future {
public:
  using completion_fn = basf::move_only_function<T() noexcept>;

  future() noexcept = default;

  explicit future(completion_fn&& on_computation_done,
                  computation_handle&& computation = {}) noexcept
      : on_computation_done_{std::move(on_computation_done)}, computation_{std::move(computation)} {
  }

  future(const future&) = delete;
  future(future&&) noexcept = default;

  future& operator=(const future&) = delete;
  future& operator=(future&& other) noexcept {
    // Note: computation_ is move assigned before the on_computation_done_ functor because
    // computation_ may depend on resources owned by on_computation_done_, and move assignment
    // may block until it is completed if the future is already active
    computation_ = std::move(other.computation_);
    on_computation_done_ = std::move(other.on_computation_done_);
    return *this;
  }

  bool available() const noexcept { return computation_.empty(); }

  void wait() noexcept { computation_.wait(); }

  T await_result() noexcept {
    SXT_DEBUG_ASSERT(on_computation_done_, "no completion set");
    this->wait();
    return on_computation_done_();
  }

  template <class F>
  auto then(F&& f) noexcept
    requires requires {
      { F{std::move(f)} } noexcept;
      f(T{});
    }
  {
    SXT_DEBUG_ASSERT(on_computation_done_, "no completion set");
    using Tp = decltype(f(std::declval<T>()));
    auto completion_p = [f = std::move(f),
                         completion = std::move(on_computation_done_)]() mutable noexcept -> Tp {
      return f(completion());
    };
    return future<Tp>{std::move(completion_p), std::move(computation_)};
  }

  template <class F>
  auto then(F&& f) noexcept
    requires requires {
      { F{std::move(f)} } noexcept;
      f();
    }
  {
    SXT_DEBUG_ASSERT(on_computation_done_, "no completion set");
    using Tp = decltype(f());
    auto completion_p = [f = std::move(f),
                         completion = std::move(on_computation_done_)]() mutable noexcept -> Tp {
      completion();
      return f();
    };
    return future<Tp>{std::move(completion_p), std::move(computation_)};
  }

private:
  completion_fn on_computation_done_;
  computation_handle computation_;
};

//--------------------------------------------------------------------------------------------------
// make_ready_future
//--------------------------------------------------------------------------------------------------
template <class T> future<T> make_ready_future(T&& value) noexcept {
  return future<T>{[res = std::move(value)]() mutable noexcept -> T { return T{std::move(res)}; }};
}

inline future<void> make_ready_future() noexcept {
  return future<void>{[]() noexcept {}};
}
} // namespace sxt::xena
