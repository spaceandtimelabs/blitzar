#pragma once

#include <coroutine>

#include "sxt/execution/async/awaiter.h"
#include "sxt/execution/async/coroutine_promise.h"
#include "sxt/execution/async/future.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// operator co_await
//--------------------------------------------------------------------------------------------------
template <class T> awaiter<T> operator co_await(future<T>&& fut) noexcept {
  return awaiter<T>{std::move(fut)};
}
} // namespace sxt::xena

//--------------------------------------------------------------------------------------------------
// coroutine_traits
//--------------------------------------------------------------------------------------------------
namespace std {
template <class T, class... Args> struct coroutine_traits<sxt::xena::future<T>, Args...> {
  using promise_type = sxt::xena::coroutine_promise<T>;
};
} // namespace std
