#include <iostream>

#include <coroutine>
#include <type_traits>

#include "sxt/execution/async/coroutine_promise.h"
#include "sxt/execution/async/future.h"

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
    auto pr = future_.promise();
    pr->set_continuation(handle.promise());
  }

  T await_resume() noexcept {
    if constexpr (std::is_same_v<T, void>) {
      return;
    } else {
      return T{std::move(future_.value())};
    }
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
