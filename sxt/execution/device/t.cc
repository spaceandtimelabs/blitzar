#include <iostream>
#include <coroutine>
#include <concepts>
#include <type_traits>
#include <utility>

#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/promise_future_base.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future
//--------------------------------------------------------------------------------------------------
/**
 * Manage asynchronous computations.
 *
 * This is a highly simplified version of a future derived from seastar
 *
 * See https://seastar.io/futures-promises/
 */
template <class T = void> class future final : protected future_base {
public:
  using value_type = T;
  using reference = std::add_lvalue_reference_t<T>;

  future() noexcept = default;

  bool ready() const noexcept { return true; }

  xena::promise<T>* promise() const noexcept {
    return static_cast<xena::promise<T>*>(static_cast<const future_base*>(this)->promise());
  }

private:
};

//--------------------------------------------------------------------------------------------------
// make_ready_future
//--------------------------------------------------------------------------------------------------
future<> make_ready_future() noexcept {
  return {};
}
} // namespace sxt::xena

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// coroutine_promise_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class Promise> class coroutine_promise_impl {
public:
  coroutine_promise_impl() noexcept = default;
  coroutine_promise_impl(const coroutine_promise_impl&) = delete;
  coroutine_promise_impl(coroutine_promise_impl&&) = delete;

  coroutine_promise_impl& operator=(const coroutine_promise_impl&) = delete;
  coroutine_promise_impl& operator=(coroutine_promise_impl&&) = delete;

  std::suspend_never initial_suspend() const noexcept { return {}; }
  std::suspend_never final_suspend() const noexcept { return {}; }

  future<T> get_return_object() noexcept { return future<T>{promise_}; }

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
    ((void)args, ...);
  }
};

template <>
class coroutine_promise<void> final
    : public detail::coroutine_promise_impl<void, coroutine_promise<void>> {
public:
  void return_void() noexcept { this->promise_.make_ready(); }
};
} // namespace sxt::xena

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// awaiter
//--------------------------------------------------------------------------------------------------
template <class T> class awaiter {
public:
  explicit awaiter(future<T>&& fut) noexcept : future_{std::move(fut)} {
  }

  bool await_ready() const noexcept { return future_.ready(); }

  template <class U>
  void await_suspend(std::coroutine_handle<coroutine_promise<U>> handle) noexcept {
    (void)handle;
  }

  T await_resume() noexcept {
    return T{};
  }

private:
  future<T> future_;
};
} // namespace sxt::xena

//--------------------------------------------------------------------------------------------------
// operator co_await
//--------------------------------------------------------------------------------------------------
namespace sxt {
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
} // namespace sxt

//--------------------------------------------------------------------------------------------------
// coroutine_traits
//--------------------------------------------------------------------------------------------------
namespace std {
template <class T, class... Args> struct coroutine_traits<sxt::xena::future<T>, Args...> {
  using promise_type = sxt::xena::coroutine_promise<T>;
};
} // namespace std

// code
using namespace sxt;

struct stream {};

xena::future<> await_stream(const stream& s) noexcept {
  return xena::make_ready_future();
}

template <class T>
xena::future<> f(const T& t) noexcept {
  stream s;
  co_await await_stream(s);
}
