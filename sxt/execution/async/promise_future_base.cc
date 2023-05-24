/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
