/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#pragma once

#include <concepts>
#include <coroutine>
#include <utility>

#include "sxt/base/error/panic.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/task.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// coroutine_promise_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class Promise> class coroutine_promise_impl : public task {
public:
  coroutine_promise_impl() noexcept = default;
  coroutine_promise_impl(const coroutine_promise_impl&) = delete;
  coroutine_promise_impl(coroutine_promise_impl&&) = delete;

  coroutine_promise_impl& operator=(const coroutine_promise_impl&) = delete;
  coroutine_promise_impl& operator=(coroutine_promise_impl&&) = delete;

  std::suspend_never initial_suspend() const noexcept { return {}; }
  std::suspend_never final_suspend() const noexcept { return {}; }

  void unhandled_exception() noexcept { baser::panic("we don't support exceptions in coroutines"); }

  future<T> get_return_object() noexcept { return future<T>{promise_}; }

  // task
  void run_and_dispose() noexcept override {
    auto handle = std::coroutine_handle<Promise>::from_promise(static_cast<Promise&>(*this));
    handle.resume();
  }

protected:
  promise<T> promise_;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// coroutine_promise
//--------------------------------------------------------------------------------------------------
template <class T>
class coroutine_promise final : public detail::coroutine_promise_impl<T, coroutine_promise<T>> {
public:
  template <class... Args>
    requires std::constructible_from<T, Args&&...>
  void return_value(Args&&... args) noexcept {
    this->promise_.set_value(std::forward<Args>(args)...);
  }
};

template <>
class coroutine_promise<void> final
    : public detail::coroutine_promise_impl<void, coroutine_promise<void>> {
public:
  void return_void() noexcept { this->promise_.make_ready(); }
};

extern template class coroutine_promise<void>;
} // namespace sxt::xena
