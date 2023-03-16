#include "sxt/execution/async/promise_future_base.h"

#include <utility>

#include "sxt/base/error/assert.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
promise_base::promise_base(promise_base&& other) noexcept
    : future_{std::exchange(other.future_, nullptr)} {
  if (future_ != nullptr) {
    future_->set_promise(this);
  }
}

future_base::future_base(future_base&& other) noexcept
    : promise_{std::exchange(other.promise_, nullptr)} {
  if (promise_ != nullptr) {
    promise_->set_future(this);
  }
}

//--------------------------------------------------------------------------------------------------
// assignment
//--------------------------------------------------------------------------------------------------
promise_base& promise_base::operator=(promise_base&& other) noexcept {
  SXT_DEBUG_ASSERT(future_ == nullptr, "promise should not have an attached future");
  future_ = std::exchange(other.future_, nullptr);
  if (future_ != nullptr) {
    future_->set_promise(this);
  }
  return *this;
}

future_base& future_base::operator=(future_base&& other) noexcept {
  SXT_DEBUG_ASSERT(promise_ == nullptr, "future should not have an attached promise");
  promise_ = std::exchange(other.promise_, nullptr);
  if (promise_ != nullptr) {
    promise_->set_future(this);
  }
  return *this;
}
} // namespace sxt::xena
