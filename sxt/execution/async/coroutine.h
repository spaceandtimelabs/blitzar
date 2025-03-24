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
#pragma once

#include <coroutine>

#include "sxt/execution/async/awaiter.h"
#include "sxt/execution/async/coroutine_promise.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/shared_future.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// operator co_await
//--------------------------------------------------------------------------------------------------
template <class T> xena::awaiter<T> operator co_await(xena::future<T>&& fut) noexcept {
  return xena::awaiter<T>{std::move(fut)};
}

template <class Fut>
  requires requires(Fut fut) {
    { xena::future<typename Fut::value_type>{std::move(fut)} } noexcept;
  }
auto operator co_await(Fut&& fut) noexcept {
  using T = typename Fut::value_type;
  return xena::awaiter<T>{xena::future<T>{std::move(fut)}};
}

template <class T> xena::awaiter<T> operator co_await(const xena::shared_future<T>& fut) noexcept {
  return xena::awaiter<T>{fut.make_future()};
}
} // namespace sxt

//--------------------------------------------------------------------------------------------------
// coroutine_traits
//--------------------------------------------------------------------------------------------------
namespace std {
template <class T, class... Args> struct coroutine_traits<sxt::xena::future<T>, Args...> {
  using promise_type = sxt::xena::coroutine_promise<T>;
};
} // namespace std
