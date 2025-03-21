#pragma once

#include <list>
#include <memory>
#include <utility>

#include "sxt/base/error/assert.h"
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
  shared_future_state(future<T>&& fut) noexcept : fut_{std::move(fut)} {
    SXT_DEBUG_ASSERT(fut_.promise() != nullptr || fut_.ready());
  }

#if 0
  shared_future_state(future<T>&& fut) noexcept
      : future_state_{fut.state()}, promise_{fut.promise()} {
    fut.set_promise(nullptr);
    promise_->set_future(nullptr);
    /* SXT_DEBUG_ASSERT(fut_.promise() != nullptr || fut_.ready()); */
  }
#endif

  shared_future_state(const shared_future_state&) = delete;
  shared_future_state(shared_future_state&&) = delete;

  shared_future_state& operator=(const shared_future_state&) = delete;
  shared_future_state& operator=(shared_future_state&&) = delete;

  future<T> get_future() noexcept {
    if (fut_.ready()) {
      if constexpr (std::is_same_v<T, void>) {
        return make_ready_future();
      } else {
        return make_ready_future<T>(fut_.value());
      }
    }
#if 0
    if (future_state_.ready()) {
      if constexpr (std::is_same_v<T, void>) {
        return make_ready_future();
      } else {
        return make_ready_future<T>(future_state_.value());
      }
    }
#endif

    /* SXT_DEBUG_ASSERT(fut_.promise() != nullptr); */
    if (promises_.empty()) {
      /* promise_->set_state(future_state_); */
      fut_.promise()->set_continuation(*this);
      keep_alive_ = this->shared_from_this();
#if 0
      SXT_DEBUG_ASSERT(fut_.promise() != nullptr);
      if constexpr (std::is_same_v<T, void>) {
        fut_.then([ptr = this->shared_from_this()]() noexcept {
          std::println(stderr, "run_and_dispose {}", (void*)ptr.get());
          ptr->run_and_dispose();
        });
      } else {
        fut_.then(
            [ptr = this->shared_from_this()](const T& val) noexcept { ptr->run_and_dispose(); });
      }
#endif
    }
    promises_.emplace_back();
    /* SXT_DEBUG_ASSERT(fut_.promise() != nullptr); */
    return future<T>{promises_.back()};
  };
private:
  void run_and_dispose() noexcept override {
    while (!promises_.empty()) {
      if constexpr (std::is_same_v<T, void>) {
        promises_.back().make_ready();
      } else {
        promises_.back().set_value(fut_.get_value());
        /* promises_.back().set_value(future_state_.value()); */
      }
      promises_.pop_back();
    }
    keep_alive_.reset();
  }

  std::shared_ptr<T> keep_alive_;
  future<T> fut_;
  /* future_state<T> future_state_; */
  
  std::list<promise<T>> promises_;
};
} // namespace sxt::xena
