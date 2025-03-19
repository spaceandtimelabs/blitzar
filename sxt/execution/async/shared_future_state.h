#pragma once

#include <list>
#include <memory>
#include <utility>

#include "sxt/execution/async/future.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// shared_future_state
//--------------------------------------------------------------------------------------------------
template <class T>
class shared_future_state final : public task,
                                  public std::enable_shared_from_this<shared_future_state<T>> {
public:
  shared_future_state() noexcept = default;
  shared_future_state(future<T>&& fut) noexcept : fut_{std::move(fut)} {}

  shared_future_state(const shared_future_state&) = delete;
  shared_future_state(shared_future_state&&) = delete;

  shared_future_state& operator=(const shared_future_state&) = delete;
  shared_future_state& operator=(shared_future_state&&) = delete;

  future<T> get_future() noexcept {
    if (fut_.ready()) {
      if constexpr (std::is_same_v<T, void>) {
        return make_ready_future<T>();
      } else {
        return make_ready_future<T>(fut_.value());
      }
    }
    if (promises_.empty()) {
      fut_.then(
          [ptr = this->shared_from_this()](const T& val) noexcept { ptr->run_and_dispose(); });
    }
    promises_.emplace_back();
    return future<T>{promises_.back()};
  };
private:
  void run_and_dispose() noexcept override {
    while (!promises_.empty()) {
      if constexpr (std::is_same_v<T, void>) {
        promises_.back().set_value();
      } else {
        promises_.back().set_value(fut_.get_value());
      }
      promises_.pop_back();
    }
  }

  future<T> fut_;
  std::list<promise<T>> promises_;
};
} // namespace sxt::xena
